# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
import lightning
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from hydra.utils import instantiate
from hydra.utils import get_method, get_class
from nequip.model.from_save import ModelFromCheckpoint
from nequip.data import AtomicDataDict
from nequip.utils import RankedLogger

import warnings
from typing import Optional, Dict


logger = RankedLogger(__name__, rank_zero_only=True)


# metrics are already synced before logging, but Lightning still sends a PossibleUserWarning about setting sync_dist=True in self.logdict()
warnings.filterwarnings(
    "ignore",
    message=".*when logging on epoch level in distributed setting to accumulate the metric across.*",
    category=PossibleUserWarning,
)


_SOLE_MODEL_KEY = "sole_model"


class NequIPLightningModule(lightning.LightningModule):
    """LightningModule for training, validating, testing and predicting with models constructed in the NequIP ecosystem.

    **Data**

    The ``NequIPLightningModule`` supports a single ``train`` dataset, but multiple ``val`` and ``test`` datasets.

    **Run Types and Metrics**

    - For ``train`` runs, users must provide ``loss`` and ``val_metrics``. The ``loss`` is computed on the training dataset to train the model, and requires each metric to have a corresponding coefficient that will be used to generate a ``weighted_sum``. This ``weighted_sum`` is the loss function that will be minimized over the course of training. ``val_metrics`` is computed on the validation dataset(s) for monitoring. Additionally, users may provide ``train_metrics`` to monitor metrics on the training dataset.
    - For ``val`` runs, users must provide ``val_metrics``.
    - For ``test`` runs, users must provide ``test_metrics``.

    **Logging Conventions**

    Logging is performed for the ``train``, ``val``, and ``test`` datasets.

    During ``train`` runs,
      * logging occurs at each batch ``step`` and at each ``epoch``,
      * there is only one training set, so no ``data_idx`` is used in the logging.

    For ``val`` and ``test`` runs,
      * logging only occurs at each validation or testing ``epoch``, i.e. one pass over the entirety of each validation/testing dataset,
      * there can be multiple validation and testing sets, so a zero-based ``data_idx`` index is used in the logging.

    Logging Format
      * ``/`` is used as a delimiter for to exploit the automatic grouping functionality of most loggers. Logged metrics will have the form ``train_{loss/metric}_{step/epoch}/{metric_name}`` and ``{val/test}{data_idx}_epoch/{metric_name}``. For example, ``train_loss_step/force_MSE``, ``train_metric_epoch/E_MAE``, ``val0_epoch/F_RMSE``, etc.
      * Note that this may have implications on how one would set the parameters for the `ModelCheckpoint <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html>`_ callback, i.e. if the name of a metric is used in the checkpoint file's name, the ``/`` will cause a directory to be created when instead a file is desired.
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
        num_datasets: Optional[Dict[str, int]] = None,
        # for caching training info
        info_dict: Optional[Dict] = None,
    ):
        super().__init__()

        # save arguments to instantiate LightningModule from checkpoint automatically
        self.save_hyperparameters()

        # === prevent `ModelFromCheckpoint` chaining ===
        # replace the `model` hparams (currently `ModelFromCheckpoint`) with the `model` hparams from the checkpoint
        # `LightningModule.load_from_checkpoint(ckpt)` will then instantiate the `LightningModule` with the original `model`
        # instead of `ModelFromCheckpoint` (which is the cause for `ModelFromCheckpoint` chaining)
        # NOTE: this forges a contract with `ModelFromCheckpoint` API, though it only assumes a form like `ModelFromCheckpoint(checkpoint_path, ...)`
        if get_method(model["_target_"]) == ModelFromCheckpoint:
            assert "checkpoint_path" in model
            self.hparams["model"] = torch.load(
                model["checkpoint_path"], map_location="cpu", weights_only=False
            )["hyper_parameters"]["model"]
            # ^ https://github.com/Lightning-AI/pytorch-lightning/blob/df5dee674243e124a2bf34d9975dd586ff008d4b/src/lightning/pytorch/core/mixins/hparams_mixin.py#L154
            # "The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable by the user. For the frozen set of initial hyperparameters, use :attr:`hparams_initial`."

        # === instantiate model ===
        model_object = self._build_model(model)

        # === account for multiple models ===
        # contract:
        # - for multiple models, they must be in the form of a `ModuleDict` of `GraphModel`s
        # - if a single `GraphModel` is provided, we wrap it in a `ModuleDict`
        # - all models must have the same `type_names`

        # the reason for `hasattr(x, "is_graph_model")` and not just `isinstance(x, GraphModel)`
        # is to support `GraphModel` from a `nequip-package`d model (see https://pytorch.org/docs/stable/package.html#torch-package-sharp-edges)
        assert isinstance(model_object, torch.nn.ModuleDict) or hasattr(
            model_object, "is_graph_model"
        )
        if not isinstance(model_object, torch.nn.ModuleDict):
            model_object = torch.nn.ModuleDict({_SOLE_MODEL_KEY: model_object})
        self.model = model_object
        type_names_list = []
        for k, v in self.model.items():
            assert hasattr(v, "is_graph_model")
            type_names_list.append(v.type_names)
            logger.debug(f"Built Model Details ({k}):\n{str(v)}")
        assert all(
            [
                all(
                    [
                        name1 == name2
                        for (name1, name2) in zip(type_names_list[0], type_names)
                    ]
                )
                for type_names in type_names_list
            ]
        ), "If multiple models are used, they must have the same type names parameter."
        type_names = type_names_list[0]  # passed to `MetricsManager`s later

        # === optimizer and lr scheduler ===
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler

        # === instantiate MetricsManager objects ===
        # must have separate MetricsManagers for each dataloader
        # num_datasets goes in order [train, val, test, predict]
        self.num_datasets = (
            num_datasets
            if num_datasets is not None
            else {
                "train": 0,
                "val": 0,
                "test": 0,
                "predict": 0,
            }
        )

        assert (
            self.num_datasets["train"] == 1
        ), "currently only support one training dataset"

        # == DDP concerns for loss ==

        # to account for loss contributions from multiple ranks later on
        # NOTE: this must be updated externally by the script that sets up the training run
        self.world_size = 1

        # == instantiate loss ==
        self.loss = instantiate(loss, type_names=type_names)
        if self.loss is not None:
            assert (
                self.loss.do_weighted_sum
            ), "`coeff` must be set for entries of the `loss` MetricsManager for a weighted sum of metrics components to be used as the loss."

        # set `dist_sync_on_step=True` for loss metrics
        # to ensure correct DDP syncing of loss function for batch steps
        for metric in self.loss.values():
            metric.dist_sync_on_step = True
        self.loss.eval()

        # == instantiate other metrics ==
        self.train_metrics = instantiate(train_metrics, type_names=type_names)
        if self.train_metrics is not None:
            self.train_metrics.eval()
        # may need to instantate multiple instances to account for multiple val and test datasets
        self.val_metrics = torch.nn.ModuleList(
            [
                instantiate(val_metrics, type_names=type_names)
                for _ in range(self.num_datasets["val"])
            ]
        )
        if self.val_metrics is not None:
            self.val_metrics.eval()
        self.test_metrics = torch.nn.ModuleList(
            [
                instantiate(test_metrics, type_names=type_names)
                for _ in range(self.num_datasets["test"])
            ]
        )
        if self.test_metrics is not None:
            self.test_metrics.eval()

        # use "/" as delimiter for loggers to automatically categorize logged metrics
        self.logging_delimiter = "/"

        # for statefulness of the run stage
        self.register_buffer("run_stage", torch.zeros((1), dtype=torch.long))

    def _build_model(self, model_config: Dict) -> torch.nn.ModuleDict:
        """Constructs a ``torch.nn.ModuleDict[str, nequip.nn.GraphModel]`` from a pure Python dictionary.

        Subclasses that require more control over how the model is built can override this method.
        """
        # reason for following implementation instead of just `hydra.utils.instantiate(model)` is to prevent omegaconf from being a model dependency
        model_config = model_config.copy()  # make a copy because of `pop` mutation
        model_builder = get_method(model_config.pop("_target_"))
        model = model_builder(**model_config)
        return model

    def configure_optimizers(self):
        """"""
        # currently support 1 optimizer and 1 scheduler
        # potentially support N optimzier and N scheduler
        # (see https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers)
        optimizer_config = self.optimizer_config.copy()
        param_groups = optimizer_config.pop(
            "param_groups",
            {"_target_": "nequip.train.lightning._default_param_group_factory"},
        )
        param_groups = instantiate(param_groups, model=self.model)
        optimizer_class = optimizer_config.pop("_target_")
        optim = get_class(optimizer_class)(params=param_groups, **optimizer_config)
        if self.lr_scheduler_config is not None:
            # instantiate lr scheduler object separately to pass the optimizer to it during instantiation
            lr_scheduler_config = dict(self.lr_scheduler_config.copy())
            scheduler = lr_scheduler_config.pop("scheduler")
            scheduler = instantiate(scheduler, optimizer=optim)
            lr_scheduler = dict(instantiate(lr_scheduler_config))
            lr_scheduler.update({"scheduler": scheduler})
            return {"optimizer": optim, "lr_scheduler": lr_scheduler}

        return optim

    def forward(self, inputs: AtomicDataDict.Type):
        """"""
        # enable grad for forces, stress, etc
        with torch.enable_grad():
            # multi-model subclasses will need to override this function
            return self.model[_SOLE_MODEL_KEY](inputs)

    @property
    def evaluation_model(self) -> torch.nn.Module:
        return self.model

    def process_target(
        self, batch: AtomicDataDict.Type, batch_idx: int, dataloader_idx: int = 0
    ) -> AtomicDataDict.Type:
        """"""
        # subclasses can override this function
        return batch.copy()

    def training_step(
        self, batch: AtomicDataDict.Type, batch_idx: int, dataloader_idx: int = 0
    ):
        """"""
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
        # In DDP training, because gradients are averaged rather than summed over nodes,
        # we get an effective factor of 1/n_rank applied to the loss. Because our loss already
        # manages correct accumulation of the metric over ranks, we want to cancel out this
        # unnecessary 1/n_rank term. If DDP is disabled, this is 1 and has no effect.
        loss = (
            loss_dict[f"train_loss_step{self.logging_delimiter}weighted_sum"]
            * self.world_size
        )
        return loss

    def on_train_epoch_end(self):
        """"""
        # optionally compute training metrics
        if self.train_metrics is not None:
            train_metric_dict = self.train_metrics.compute(
                prefix=f"train_metric_epoch{self.logging_delimiter}"
            )
            self.log_dict(train_metric_dict)
            self.train_metrics.reset()
        # loss
        loss_dict = self.loss.compute(
            prefix=f"train_loss_epoch{self.logging_delimiter}"
        )
        self.log_dict(loss_dict)
        self.loss.reset()

    def validation_step(
        self, batch: AtomicDataDict.Type, batch_idx: int, dataloader_idx: int = 0
    ):
        """"""
        target = self.process_target(batch, batch_idx, dataloader_idx)

        # === update basic val metrics ===
        output = self(batch)
        with torch.no_grad():
            metric_dict = self.val_metrics[dataloader_idx](
                output,
                target,
                prefix=f"val{dataloader_idx}_step{self.logging_delimiter}",
            )

        return metric_dict

    def on_validation_epoch_end(self):
        """"""
        # === reset basic val metrics ===
        for idx, metrics in enumerate(self.val_metrics):
            metric_dict = metrics.compute(
                prefix=f"val{idx}_epoch{self.logging_delimiter}"
            )
            self.log_dict(metric_dict)
            metrics.reset()

    def test_step(
        self, batch: AtomicDataDict.Type, batch_idx: int, dataloader_idx: int = 0
    ):
        """"""
        target = self.process_target(batch, batch_idx, dataloader_idx)

        # === update basic test metrics ===
        output = self(batch)
        with torch.no_grad():
            metric_dict = self.test_metrics[dataloader_idx](
                output,
                target,
                prefix=f"test{dataloader_idx}_step{self.logging_delimiter}",
            )
        metric_dict.update({f"test_{dataloader_idx}_output": output})

        return metric_dict

    def on_test_epoch_end(self):
        """"""
        # === reset basic test metrics ===
        for idx, metrics in enumerate(self.test_metrics):
            metric_dict = metrics.compute(
                prefix=f"test{idx}_epoch{self.logging_delimiter}"
            )
            self.log_dict(metric_dict)
            metrics.reset()


def _default_param_group_factory(model):
    return model.parameters()
