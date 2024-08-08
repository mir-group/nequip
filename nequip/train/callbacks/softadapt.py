import numpy as np

import lightning
from lightning.pytorch.callbacks import Callback

from nequip.data import AtomicDataDict
from nequip.train import NequIPLightningModule

from typing import List


class SoftAdapt(Callback):
    """Adaptively modify loss coefficients over a training run using the `SoftAdapt <https://arxiv.org/abs/2403.18122>`_ scheme.

    Example usage in config where the loss coefficients are updated every 5 epochs:
    ::

        callbacks:
          - _target_: nequip.train.callbacks.SoftAdapt
            beta: 1.1
            interval: epoch
            frequency: 5

    Args:
        beta (float): ``SoftAdapt`` hyperparameter (see paper)
        interval (str): ``batch`` or ``epoch``
        frequency (int): number of intervals between loss coefficient updates
        eps (float): small value to avoid division by zero
    """

    def __init__(
        self,
        beta: float,
        interval: str,
        frequency: int,
        eps: float = 1e-8,
    ):
        assert interval in ["batch", "epoch"]
        assert frequency >= 1

        self.beta = beta
        self.interval = interval
        self.frequency = frequency
        self.eps = eps

        self.prev_losses = None
        self.cached_coeffs = []

    def _softadapt_update(
        self,
        new_losses: List[float],
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ):
        if self.interval == "epoch":
            step = trainer.current_epoch  # use epochs
        else:
            step = trainer.global_step  # use batches

        # empty list of cached weights to store for next cycle
        if step % self.frequency == 0:
            self.cached_coeffs = []

        # compute and cache new loss weights over the update cycle
        if self.prev_losses is None:
            self.prev_losses = new_losses
            return
        else:
            # compute normalized loss change
            loss_change = new_losses - self.prev_losses
            loss_change = loss_change / np.maximum(
                np.linalg.norm(loss_change), self.eps
            )
            self.prev_losses = new_losses
            # compute new weights with softmax
            exps = np.exp(self.beta * loss_change)
            self.cached_coeffs.append(exps / (np.sum(exps) + self.eps))

        # average weights over previous cycle and update
        if step % self.frequency == 1:
            softadapt_weights = np.mean(np.stack(self.cached_coeffs, axis=-1), axis=-1)
            # make sure None entries stay None
            new_coeffs = []
            for idx, new_coeff in enumerate(softadapt_weights.tolist()):
                new_coeffs.append(
                    new_coeff if pl_module.loss.coeffs[idx] is not None else None
                )
            pl_module.loss.set_coeffs(new_coeffs)

    def on_train_batch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
        batch: AtomicDataDict.Type,
        batch_idx: int,
    ):
        """"""
        if trainer.global_step == 0:
            return
        if self.interval == "batch":
            new_losses = np.array(pl_module.loss.metrics_values_step)
            self._softadapt_update(new_losses, trainer, pl_module)

    def on_train_epoch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ):
        """"""
        if trainer.current_epoch == 0:
            return
        if self.interval == "epoch":
            new_losses = np.array(pl_module.loss.metrics_values_epoch)
            self._softadapt_update(new_losses, trainer, pl_module)

    def state_dict(self):
        """"""
        return {
            "beta": self.beta,
            "interval": self.interval,
            "frequency": self.frequency,
            "eps": self.eps,
            "prev_losses": self.prev_losses,
            "cached_coeffs": self.cached_coeffs,
        }

    def load_state_dict(self, state_dict):
        """"""
        self.beta = state_dict["beta"]
        self.interval = state_dict["interval"]
        self.frequency = state_dict["frequency"]
        self.eps = state_dict["eps"]
        self.prev_losses = state_dict["prev_losses"]
        self.cached_coeffs = state_dict["cached_coeffs"]
