# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from math import sqrt, exp
import lightning
from lightning.pytorch.callbacks import Callback
from nequip.data import AtomicDataDict
from nequip.train import NequIPLightningModule
from typing import List, Dict


class SoftAdapt(Callback):
    """Adaptively modify loss coefficients over a training run using the `SoftAdapt <https://arxiv.org/abs/2403.18122>`_ scheme.

    .. warning::
        The SoftAdapt requires that all components of the loss function contribute to the loss function, i.e. that their ``coeff`` in the ``MetricsManager`` is not ``None``.

    .. warning::
        It is dangerous to restart training (with SoftAdapt) and use a differently configured loss function for the restart because SoftAdapt's loaded checkpoint state will become ill-suited for the new loss function.

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

        self.prev_losses: Dict[str, float] = None
        self.cached_coeffs: List[Dict[str, float]] = []

    def _softadapt_update(
        self,
        new_losses: Dict[str, float],
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ):
        # === sanity checks ===
        assert all(
            [
                metric_dict["coeff"] is not None
                for metric_dict in pl_module.loss.metrics.values()
            ]
        ), "all components of loss must have `coeff!=None` to use the SoftAdapt callback"

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
            # TODO (maybe): the check could be stronger by matching the keys themselves, but might add overhead
            assert len(new_losses) == len(self.prev_losses)

            # compute loss component changes
            loss_changes = {
                k: new_losses[k] - self.prev_losses[k] for k in new_losses.keys()
            }
            # normalize and apply softmax
            sum_of_squares = sum(
                [loss_changes[k] * loss_changes[k] for k in new_losses.keys()]
            )
            factor = self.beta / max(sqrt(sum_of_squares), self.eps)
            exps = {k: exp(factor * v) for k, v in loss_changes.items()}
            softmax_denom = sum([exps[k] for k in new_losses.keys()]) + self.eps
            new_coeffs = {k: exp_term / softmax_denom for k, exp_term in exps.items()}

            # update with new coefficients
            self.cached_coeffs.append(new_coeffs)
            del new_coeffs
            # update previous loss components
            self.prev_losses = new_losses

        # average weights over previous cycle and update
        if step % self.frequency == 1:
            num_updates = len(self.cached_coeffs)
            softadapt_weights = {
                metric_name: sum(
                    [self.cached_coeffs[idx][metric_name] for idx in range(num_updates)]
                )
                / num_updates
                for metric_name in pl_module.loss.keys()
            }
            pl_module.loss.set_coeffs(softadapt_weights)

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
            self._softadapt_update(
                pl_module.loss.metrics_values_step, trainer, pl_module
            )

    def on_train_epoch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ):
        """"""
        if trainer.current_epoch == 0:
            return
        if self.interval == "epoch":
            self._softadapt_update(
                pl_module.loss.metrics_values_epoch, trainer, pl_module
            )

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
