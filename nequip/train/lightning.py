import torch
import lightning
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from hydra.utils import instantiate
from hydra.utils import get_method
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


class NequIPLightningModule(lightning.LightningModule):
    """LightningModule for training, validating, testing and predicting with models constructed in the NequIP ecosystem.

    Data
    ####
    The ``NequIPLightningModule`` supports a single ``train`` dataset, but multiple ``val`` and ``test`` datasets.

    Run Types and Metrics
    #####################
    - For ``train`` runs, users must provide ``loss`` and ``val_metrics``. The ``loss`` is computed on the training dataset to train the model, and requires each metric to have a corresponding coefficient that will be used to generate a ``weighted_sum``. This ``weighted_sum`` is the loss function that will be minimized over the course of training. ``val_metrics`` is computed on the validation dataset(s) for monitoring. Additionally, users may provide ``train_metrics`` to monitor metrics on the training dataset.
    - For ``val`` runs, users must provide ``val_metrics``.
    - For ``test`` runs, users must provide ``test_metrics``.

    Logging Conventions
    ###################
    Logging is performed for the ``train``, ``val``, and ``test`` datasets.

    During ``train`` runs,
      * logging occurs at each batch ``step`` and at each ``epoch``,
      * there is only one training set, so no ``data_idx`` is used in the logging.

    For ``val`` and ``test`` runs,
      * logging only occurs at each validation or testing ``epoch``, i.e. one pass over the entirety of each validation/testing dataset,
      * there can be multiple validation and testing sets, so a zero-based ``data_idx`` index is used in the logging.

    Logging Format
      * ``/`` is used as a delimiter for to exploit the automatic grouping functionality of most loggers. Logged metrics will have the form ``train_{loss/metric}_{step/epoch}/{metric_name}`` and ``{val/test}{data_idx}_epoch/{metric_name}``. For example, ``train_loss_step/force_MSE``, ``train_metric_epoch/E_MAE``, ``val0_epoch/F_RMSE``, etc.
      * Note that this may have implications on how one would set the parameters for the ``ModelCheckpoint`` callback, i.e. if the name of a metric is used in the checkpoint file's name, the ``/`` will cause a directory to be created when instead a file is desired.

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
        super().__init__()
        self.save_hyperparameters()

        # === instantiate model ===
        # reason for following implementation instead of just `hydra.utils.instantiate(model)` is to prevent omegaconf from being a model dependency
        model = model.copy()  # make a copy because of `pop`
        model_builder = get_method(model.pop("_target_"))
        self.model = model_builder(**model)

        logger.debug(f"Built Model Details:\n{str(self.model)}")
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler

        # === instantiate MetricsManager objects ===
        # must have separate MetricsManagers for each dataloader
        # num_datasets goes in order [train, val, test, predict]
        self.num_datasets = kwargs.get(
            "num_datasets",
            {
                "train": 0,
                "val": 0,
                "test": 0,
                "predict": 0,
            },
        )

        assert (
            self.num_datasets["train"] == 1
        ), "currently only support one training dataset"

        # == DDP concerns for loss ==

        # to account for loss contributions from multiple ranks later on
        # NOTE: this must be updated externally by the script that sets up the training run
        self.world_size = 1

        # add dist_sync_on_step for loss metrics
        for metric_dict in loss["metrics"]:
            # silently ensure that dist_sync_on_step is true for loss metrics
            metric_dict["metric"]["dist_sync_on_step"] = True

        # == instantiate loss ==
        self.loss = instantiate(loss, type_names=self.model.type_names)
        if self.loss is not None:
            assert (
                self.loss.do_weighted_sum
            ), "`coeff` must be set for entries of the `loss` MetricsManager for a weighted sum of metrics components to be used as the loss."

        # == instantiate other metrics ==
        self.train_metrics = instantiate(
            train_metrics, type_names=self.model.type_names
        )
        # may need to instantate multiple instances to account for multiple val and test datasets
        self.val_metrics = torch.nn.ModuleList(
            [
                instantiate(val_metrics, type_names=self.model.type_names)
                for _ in range(self.num_datasets["val"])
            ]
        )
        self.test_metrics = torch.nn.ModuleList(
            [
                instantiate(test_metrics, type_names=self.model.type_names)
                for _ in range(self.num_datasets["test"])
            ]
        )

        # use "/" as delimiter for loggers to automatically categorize logged metrics
        self.logging_delimiter = "/"

        # for statefulness of the run stage
        self.register_buffer("run_stage", torch.zeros((1), dtype=torch.long))

    def configure_optimizers(self):
        """"""
        # currently support 1 optimizer and 1 scheduler
        # potentially support N optimzier and N scheduler
        # (see https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers)
        optim = instantiate(self.optimizer_config, params=self.model.parameters())
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
            return self.model(inputs)

    def training_step(
        self, batch: AtomicDataDict.Type, batch_idx: int, dataloader_idx: int = 0
    ):
        """"""
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
        target = batch.copy()
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
        target = batch.copy()
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
        for idx, metrics in enumerate(self.test_metrics):
            metric_dict = metrics.compute(
                prefix=f"test{idx}_epoch{self.logging_delimiter}"
            )
            self.log_dict(metric_dict)
            metrics.reset()
