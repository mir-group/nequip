# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import lightning
from lightning.pytorch.callbacks import Callback
from nequip.train import NequIPLightningModule
from typing import Dict


class LossCoefficientScheduler(Callback):
    """Schedule loss coefficients during training.

    The ``LossCoefficientScheduler`` takes a single argument ``schedule``, which is a ``Dict[int, Dict[str, float]]`` where the keys are the epochs at which the loss coefficients change and the values are dictionaries mapping loss metric names (corresponding to how the loss was configured) to their coefficients.

    When the trainer's epoch counter matches any of the keys (representing epochs), the loss coefficients will be changed to the values (representing the coefficients for each loss term).

    The coefficients will be normalized to sum up to 1 in line with the convention of ``MetricsManager``.

    Example usage in config where there are two loss contributions:
    ::

        callbacks:
          - _target_: nequip.train.callbacks.LossCoefficientScheduler
            schedule:
              100:
                per_atom_energy_mse: 1.0
                forces_mse: 5.0
              200:
                per_atom_energy_mse: 5.0
                forces_mse: 1.0

    Args:
        schedule (Dict[int, Dict[str,float]]): map of epoch to loss coefficient dictionary
    """

    def __init__(self, schedule: Dict[int, Dict[str, float]]):
        # ensure that the keys are `int`s
        self.schedule = {int(k): v for k, v in schedule.items()}
        # sanity check - epochs are >= 0
        assert all([epoch >= 0 for epoch in self.schedule.keys()])

    def on_train_epoch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ) -> None:
        """"""
        # only change loss coefficients at the designated epochs
        if trainer.current_epoch not in self.schedule.keys():
            return
        # set the loss coefficients
        pl_module.loss.set_coeffs(self.schedule[trainer.current_epoch])
