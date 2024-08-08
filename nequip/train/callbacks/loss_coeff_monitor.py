import torch
import lightning
from lightning.pytorch.callbacks import Callback

from nequip.data import AtomicDataDict
from nequip.train import NequIPLightningModule


class LossCoefficientMonitor(Callback):
    """Monitor and log loss coefficients during training.

    Example usage in config to log loss coefficients every 5 epochs:
    ::

        callbacks:
          - _target_: nequip.train.callbacks.LossCoefficientMonitor
            interval: epoch
            frequency: 5

    Args:
        interval (str): ``batch`` or ``epoch``
        frequency (int): number of intervals between each instance of loss coefficient logging
    """

    def __init__(
        self,
        interval: str,
        frequency: int,
    ):
        assert interval in ["batch", "epoch"]
        assert frequency >= 1
        self.interval = interval
        self.frequency = frequency

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
        outputs: torch.Tensor,
        batch: AtomicDataDict.Type,
        batch_idx: int,
    ) -> None:
        """"""
        if self.interval == "batch":
            if trainer.global_step % self.frequency == 0:
                loss = pl_module.loss
                for idx in range(loss.num_metrics):
                    if loss.coeffs[idx] is not None:
                        pl_module.log(loss.names[idx] + "_coeff", loss.coeffs[idx])

    def on_train_epoch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ) -> None:
        """"""
        if self.interval == "epoch":
            if trainer.current_epoch % self.frequency == 0:
                loss = pl_module.loss
                for idx in range(loss.num_metrics):
                    if loss.coeffs[idx] is not None:
                        pl_module.log(loss.names[idx] + "_coeff", loss.coeffs[idx])
