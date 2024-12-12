import torch
from .lightning import NequIPLightningModule

from nequip.data import AtomicDataDict

from itertools import accumulate
from typing import Dict, Optional


class ConFIGLightningModule(NequIPLightningModule):
    """
    Conflict-free inverse gradient (ConFIG) approach to multitask learning. See https://arxiv.org/abs/2408.11104.

    The arguments for this class are exactly the same as ``NequIPLightningModule``, but the loss coefficients take on a different meaning -- they are now the "b" in the "Ax=b" linear solve (see paper).

    Note:
      LR schedulers won't be able to monitor training metrics using this class -- which should not be a problem since LR schedulers should usually be monitoring validation metrics.
    """

    def __init__(
        self,
        model: Dict,
        optimizer: Optional[Dict] = None,
        lr_scheduler: Optional[Dict] = None,
        loss: Optional[Dict] = None,
        train_metrics: Optional[Dict] = None,
        val_metrics: Optional[Dict] = None,
        test_metrics: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss=loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            **kwargs,
        )
        # see https://lightning.ai/docs/pytorch/stable/common/optimization.html#id2
        self.automatic_optimization = False

        # === process model params ===
        # ingredients to do the reverse of torch.cat([torch.flatten(),...])
        param_dict = dict(self.model.named_parameters())
        self.ConFIG_model_param_names = list(param_dict.keys())
        self.ConFIG_param_numel_list = [
            param_dict[k].numel() for k in self.ConFIG_model_param_names
        ]
        self.ConFIG_param_batch_list = [
            0,
        ] + list(accumulate(self.ConFIG_param_numel_list))
        self.ConFIG_param_shape_list = [
            param_dict[k].shape for k in self.ConFIG_model_param_names
        ]
        del param_dict

        # === process loss functions ===
        self.ConFIG_loss_idxs = [
            idx
            for idx in range(self.loss.num_metrics)
            if self.loss.coeffs[idx] is not None
        ]
        assert (
            len(self.ConFIG_loss_idxs) >= 1
        ), f"`ConFIGLightningModule` is used for cases with multiple loss components, but {len(self.ConFIG_loss_idxs)} found"
        self.ConFIG_loss_component_keys = [
            f"train_loss_step{self.logging_delimiter}" + self.loss.names[idx]
            for idx in self.ConFIG_loss_idxs
        ]

        # === method specific hyperparameters ===
        # TODO: does eps need a model_dtype dependence?
        # TODO: should this be a user-controlled hyperparameter?
        self.ConFIG_eps = 1e-8  # hardcode for now

    def training_step(
        self, batch: AtomicDataDict.Type, batch_idx: int, dataloader_idx: int = 0
    ):
        """"""
        # we have to override the training step to perform the backwards manually
        # everything looks the same except the end of `training_step`

        target = batch.copy()
        output = self(batch)

        # optionally compute training metrics
        if self.train_metrics is not None:
            with torch.no_grad():
                train_metric_dict = self.train_metrics(
                    output, target, prefix=f"train_metric_step{self.logging_delimiter}"
                )
            self.log_dict(train_metric_dict)

        # compute loss and return
        loss_dict = self.loss(
            output, target, prefix=f"train_loss_step{self.logging_delimiter}"
        )
        self.log_dict(loss_dict)

        # === manual optimization part ===
        param_dict = dict(self.model.named_parameters())
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        # === get loss component gradients ===
        loss_component_grads = []
        for idx, loss_component_key in enumerate(self.ConFIG_loss_component_keys):
            # get loss component and backward (with world size factor for DDP)
            loss_component = loss_dict[loss_component_key] * self.world_size
            # gradients should get synced by DDP during the below backwards pass,
            # such that afterwards we have gradients of the loss term w.r.t. the global batch.
            self.manual_backward(
                loss_component, retain_graph=idx < len(self.ConFIG_loss_idxs) - 1
            )
            # ^ don't retain graph for the last term

            # collect gradients and assemble
            grads = torch.cat(
                [param_dict[k].grad.flatten() for k in self.ConFIG_model_param_names]
            )
            loss_component_grads.append(grads)
            del grads  # free some memory

            # set grads to None before next loss component
            # this implementation is more transparent than `opt.zero_grad()`
            for k in self.ConFIG_model_param_names:
                param_dict[k].grad = None

        # === set up and perform linear solve ===
        # NOTE: the following lines correspond to Eqs 2 and 3 of the paper

        # construct normalized gradient matrix (num_loss_components, stacked_gradient_dim)
        A_raw = torch.stack(loss_component_grads, dim=0)
        A = torch.nn.functional.normalize(A_raw, dim=1, eps=self.ConFIG_eps)
        # construct loss coeff vector (num_loss_components,)
        b = torch.nn.functional.normalize(
            torch.tensor(
                [self.loss.coeffs[idx] for idx in self.ConFIG_loss_idxs],
                dtype=A.dtype,
                device=A.device,
            ),
            dim=0,
            eps=self.ConFIG_eps,
        )
        # ^ do it here to futureproof for possibility of changing the coeffs over training

        # linear solve and normalize
        x = torch.linalg.lstsq(A, b).solution
        x = torch.nn.functional.normalize(x, dim=0, eps=self.ConFIG_eps)

        # construct the gradient vector
        # note that the unnormalized A is used
        new_grad = torch.sum(A_raw * x) * x

        # === set gradients ===
        # populate param grads with conflict-free gradients
        for idx, name in enumerate(self.ConFIG_model_param_names):
            grad_entry = torch.narrow(
                new_grad,
                0,
                self.ConFIG_param_batch_list[idx],
                self.ConFIG_param_numel_list[idx],
            )
            param_dict[name].grad = grad_entry.view(self.ConFIG_param_shape_list[idx])

        # === finally take step ===
        opt.step()

        # apply the DDP loss rescale, as usual (we already did this for the individual loss terms above before calling  `manual_backward` on them)
        loss = (
            loss_dict[f"train_loss_step{self.logging_delimiter}weighted_sum"]
            * self.world_size
        )
        return loss
