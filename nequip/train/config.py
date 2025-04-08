# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from .lightning import NequIPLightningModule
from .ema import EMALightningModule

from nequip.data import AtomicDataDict
from nequip.utils.versions import _TORCH_GE_2_6
from itertools import accumulate

from typing import Dict, Optional


class ConFIGLightningModule(NequIPLightningModule):
    """
    Conflict-free inverse gradient (ConFIG) approach to multitask learning. See https://arxiv.org/abs/2408.11104.

    The arguments for this class are exactly the same as ``NequIPLightningModule``, but the loss coefficients take on a different meaning -- they are now the "b" in the "Ax=b" linear solve (see paper).

    Set ``lsqr=False`` to use the pseudo-inverse of the gradient matrix to determine the update direction (instead of the default least squares method), as certain devices may not be able to do the (underdetermined) least squares solve (e.g. ROCm).

    Note:
        Only ``ReduceLROnPlateau`` works with this class. The following warning may be safely ignored.
        ``The lr scheduler dict contains the key(s) ['monitor'], but the keys will be ignored. You need to call `lr_scheduler.step()` manually in manual optimization.``

    Note:
        LR schedulers won't be able to monitor training metrics using this class -- which should not be a problem since LR schedulers should usually be monitoring validation metrics.

    Note:
        To use gradient clipping in training, the ``gradient_clip_val`` must be provided to this training module and not to ``Trainer``, as automatic gradient clipping is not supported for manual optimization with PyTorch.

    Args:
        gradient_clip_val (Union[int, float, None]): gradient clipping value (default: ``None``, which disables gradient clipping)
        gradient_clip_algorithm (Optional[str]): ``value`` to clip by value, or ``norm`` to clip by norm (default: ``norm``)
        lsqr (bool): whether to use least squares solve for determining best update direction (default: ``True``)
        norm_eps (float): small value to avoid division by zero during normalization (default: ``1e-8``)
    """

    def __init__(
        self,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
        lsqr: bool = True,
        norm_eps: float = 1e-8,
        **kwargs,
    ):
        # === torch>=2.6 requires the flag for compile with multiple backwards ===
        if _TORCH_GE_2_6:
            # relevant PyTorch commit: https://github.com/pytorch/pytorch/commit/87059d4547551f197731f5c084e3be6054797578
            # comments from PyTorch code:
            # This controls whether we collect donated buffer. This flag must be set
            # False if a user wants to retain_graph=True for backward.
            torch._functorch.config.donated_buffer = False

        super().__init__(**kwargs)
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
        assert (
            len(self.loss) >= 1
        ), f"`ConFIGLightningModule` is used for cases with multiple loss components, but {len(self.loss)} found"
        self.ConFIG_loss_component_keys: Dict[str, str] = {
            metric_name: f"train_loss_step{self.logging_delimiter}" + metric_name
            for metric_name in self.loss.keys()
        }

        # === method specific hyperparameters ===
        self.ConFIG_eps = norm_eps
        self.ConFIG_lsqr = lsqr

        # temporary narrow solution to accommodate only ReduceLROnPlateau LR scheduling
        if self.lr_scheduler_config is not None:
            scheduler = self.lr_scheduler_config["scheduler"]["_target_"]
            assert (
                "ReduceLROnPlateau" in scheduler
            ), f"only `ReduceLROnPlateau` LR scheduler is usable with `ConFIGLightningModule` but found `{scheduler}`"
            monitor = self.lr_scheduler_config["monitor"]
            assert (
                "val" in monitor
            ), f"Only validation metrics can be monitored for LR scheduling with `ConFIGLightningModule`, but found {monitor}"
            assert (
                self.lr_scheduler_config["interval"] == "epoch"
            ), "only `interval=epoch` allowed for LR scheduling with `ConFIGLightningModule`"
            self.ConFIG_monitor = monitor

        # gradient clipping
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

    def training_step(
        self, batch: AtomicDataDict.Type, batch_idx: int, dataloader_idx: int = 0
    ):
        """"""
        # we have to override the training step to perform the backwards manually
        # the details are in encapsulated in `_ConFIG_training_step`
        loss = self._ConFIG_training_step(batch, batch_idx, dataloader_idx)
        return loss

    def _ConFIG_training_step(
        self, batch: AtomicDataDict.Type, batch_idx: int, dataloader_idx: int = 0
    ):
        # everything looks the same as the base NequIPLightningModule up to the `manual_optimization` part
        target = self.process_target(batch, batch_idx, dataloader_idx)
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

        # apply the DDP loss rescale
        loss = (
            loss_dict[f"train_loss_step{self.logging_delimiter}weighted_sum"]
            * self.world_size
        )

        # === manual optimization part ===
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        # === peform "backwards" for ConFIG ===
        self._ConFIG_backwards(loss_dict)

        # === gradient clipping ===
        self.clip_gradients(
            opt,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
        )

        # === finally take step ===
        opt.step()
        return loss

    def _ConFIG_backwards(self, loss_dict):

        # account for loss contributions that can be `None`
        coeff_dict: Dict[str, float] = {
            metric_name: metric_dict["coeff"]
            for metric_name, metric_dict in self.loss.metrics.items()
        }
        num_active_loss_components = sum(
            [coeff is not None for coeff in coeff_dict.values()]
        )
        if num_active_loss_components == 0:
            raise RuntimeError(
                "At least one active loss component is required for training, i.e. at least on component in the loss function must have `coeff` that is not `None`."
            )
        elif num_active_loss_components == 1:
            # short-circuit if there's only one active loss component
            self.manual_backward(
                (
                    loss_dict[f"train_loss_step{self.logging_delimiter}weighted_sum"]
                    * self.world_size
                )
            )
            return

        # if we get here, it means there's at least two active loss components and we proceed to the ConFIG logic
        param_dict = dict(self.model.named_parameters())
        # === get loss component gradients ===
        loss_component_grads = []
        loss_component_coefficients = []
        for idx, metric_name in enumerate(self.loss.keys()):
            # skip the loop if there's no coefficient
            if coeff_dict[metric_name] is None:
                continue
            loss_component_coefficients.append(coeff_dict[metric_name])
            # get loss component and backward (with world size factor for DDP)
            loss_component_key = self.ConFIG_loss_component_keys[metric_name]
            loss_component = loss_dict[loss_component_key] * self.world_size
            # gradients should get synced by DDP during the below backwards pass,
            # such that afterwards we have gradients of the loss term w.r.t. the global batch.
            self.manual_backward(
                loss_component, retain_graph=idx < num_active_loss_components - 1
            )
            # ^ don't retain graph for the last term

            # collect gradients and assemble
            # note that grads will be in the highest dtype, i.e. float64 if there are any float64 grad components
            grads = torch.cat(
                [
                    (
                        param_dict[k].grad.flatten()
                        if param_dict[k].grad is not None
                        else torch.zeros_like(param_dict[k]).flatten()
                    )
                    for k in self.ConFIG_model_param_names
                ]
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
                loss_component_coefficients,
                dtype=A.dtype,
                device=A.device,
            ),
            dim=0,
            eps=self.ConFIG_eps,
        )
        # ^ do it here to futureproof for possibility of changing the coeffs over training

        # linear solve and normalize
        if self.ConFIG_lsqr:
            x = torch.linalg.lstsq(A, b).solution
        else:  # equivalent but slower least-squares solve:
            x = torch.linalg.pinv(A) @ b
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
            # set to correct dtype before assigning
            param_dict[name].grad = grad_entry.to(dtype=param_dict[name].dtype).view(
                self.ConFIG_param_shape_list[idx]
            )

    def on_validation_epoch_end(self):
        """"""
        # === reset basic val metrics ===
        for idx, metrics in enumerate(self.val_metrics):
            metric_dict = metrics.compute(
                prefix=f"val{idx}_epoch{self.logging_delimiter}"
            )
            self.log_dict(metric_dict)
            metrics.reset()

        # === ReduceLROnPlateau scheduler ===
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step(metric_dict[self.ConFIG_monitor])


# note the inheritance order -- this means `EMALightningModule` takes precedence over `ConFIGLightningModule` as a parent
class EMAConFIGLightningModule(EMALightningModule, ConFIGLightningModule):
    """Composition of ``ConFIGLightningModule`` and ``EMALightningModule``

    Args:
        gradient_clip_val (Union[int, float, None]): gradient clipping value (default: ``None``, which disables gradient clipping)
        gradient_clip_algorithm (Optional[str]): ``value`` to clip by value, or ``norm`` to clip by norm (default: ``norm``)
        lsqr (bool): whether to use least squares solve for determining best update direction (default: ``True``)
        norm_eps (float): small value to avoid division by zero during normalization (default: ``1e-8``)
        ema_decay (float): decay constant for the exponential moving average (EMA) of model weights (default ``0.999``)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # some sanity checks that diamond inheritance worked correctly
        assert not self.automatic_optimization

    def training_step(
        self, batch: AtomicDataDict.Type, batch_idx: int, dataloader_idx: int = 0
    ):
        """"""
        # we have to override the training step to perform the backwards manually
        # the details are in encapsulated in `_ConFIG_training_step`
        loss = self._ConFIG_training_step(batch, batch_idx, dataloader_idx)

        # since we're using `manual_optimization`, `self.optimizer_step` is not used
        # so we update the EMA weights here
        self.ema.update_parameters(self.model)
        return loss
