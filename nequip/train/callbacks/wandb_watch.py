# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import lightning
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger

from nequip.train import NequIPLightningModule


class WandbWatch(Callback):
    """Monitor and log weights and gradients during training with PyTorch Lightning's ``WandbLogger``.

    This class provides a way to call https://docs.wandb.ai/ref/python/watch/ when using a ``WandbLogger`` for monitoring weights and gradients over the course of training.

    Args:
        log_freq (int): frequency (in batches) to log gradients and parameters
        log_type (str): specifies whether to log ``"gradients"``, ``"parameters"``, or ``"all"``
        log_graph (bool): whether to log the model's computational graph
    """

    def __init__(
        self,
        log_freq: int,
        log: str = "gradients",
        log_graph: bool = False,
    ):
        self.log_freq = log_freq
        assert log in ["gradients", "parameters", "all", None]
        self.log_type = log
        self.log_graph = log_graph

    def on_train_start(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ) -> None:
        """"""
        assert isinstance(
            trainer.logger, WandbLogger
        ), "NequIP's `WandbWatch` callback only works for `WandbLogger` loggers"
        # see https://docs.wandb.ai/ref/python/watch/
        trainer.logger.watch(
            pl_module.model,
            log=self.log_type,
            log_freq=self.log_freq,
            log_graph=self.log_graph,
        )

    def on_train_end(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ) -> None:
        """"""
        # see unwatch syntax
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#lightning.pytorch.loggers.wandb.WandbLogger
        trainer.logger.experiment.unwatch(pl_module.model)
