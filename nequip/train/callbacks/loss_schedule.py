from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

from nequip.train import Trainer, Loss

# Making this a dataclass takes care of equality operators, handing restart consistency checks


@dataclass
class SimpleLossSchedule:
    """Schedule `loss_coeffs` through a training run.

    To use this in a training, set in your YAML file:

        start_of_epoch_callbacks:
         - !!python/object:nequip.train.callbacks.loss_schedule.SimpleLossSchedule {"schedule": [[30, {"forces": 1.0, "total_energy": 0.0}], [30, {"forces": 0.0, "total_energy": 1.0}]]}

    This funny syntax tells PyYAML to construct an object of this class.

    Each entry in the schedule is a tuple of the 1-based epoch index to start that loss coefficient set at, and a dict of loss coefficients.
    """

    schedule: List[Tuple[int, Dict[str, float]]] = None

    def __call__(self, trainer: Trainer):
        assert (
            self in trainer._start_of_epoch_callbacks
        ), "must be start not end of epoch"
        # user-facing 1 based indexing of epochs rather than internal zero based
        iepoch: int = trainer.iepoch + 1
        if iepoch < 1:  # initial validation epoch is 0 in user-facing indexing
            return
        loss_function: Loss = trainer.loss

        assert self.schedule is not None
        schedule_start_epochs = np.asarray([e[0] for e in self.schedule])
        # make sure they are ascending
        assert len(schedule_start_epochs) >= 1
        assert schedule_start_epochs[0] >= 2, "schedule must start at epoch 2 or later"
        assert np.all(
            (schedule_start_epochs[1:] - schedule_start_epochs[:-1]) > 0
        ), "schedule start epochs must be strictly ascending"
        # we are running at _start_ of epoch, so we need to apply the right change for the current epoch
        current_change_idex = np.searchsorted(schedule_start_epochs, iepoch + 1) - 1
        # ^ searchsorted 3 in [2, 10, 19] would return 1, for example
        # but searching 2 in [2, 10, 19] gives 0, so we actually search iepoch + 1 to always be ahead of the start
        # apply the current change to handle restarts
        if current_change_idex >= 0:
            new_coeffs = self.schedule[current_change_idex][1]
            assert (
                loss_function.coeffs.keys() == new_coeffs.keys()
            ), "all coeff schedules must contain all loss terms"
            loss_function.coeffs.update(new_coeffs)
