# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import lightning
from lightning.pytorch.callbacks import Callback
from nequip.train import NequIPLightningModule
from typing import Dict


class LossCoefficientScheduler(Callback):
    """Schedule loss coefficients during training.

    The ``LossCoefficientScheduler`` takes a single argument ``schedule``, which is a ``Dict[int, Dict[str, float]]`` where the keys are the epochs at which the loss coefficients change and the values are dictionaries mapping loss metric names (corresponding to how the loss was configured) to their coefficients.

    When the trainer's epoch counter matches any of the keys (representing epochs), the loss coefficients will be changed to the values (representing the coefficients for each loss term).

    The coefficients will be normalized to sum up to 1 in line with the convention of :class:`~nequip.train.MetricsManager`.

    Example usage in config where there are two loss contributions:

    .. code-block:: yaml

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


class LinearLossCoefficientScheduler(Callback):
    """Linearly schedule loss coefficients during training.

    The ``LinearLossCoefficientScheduler`` linearly interpolates loss coefficients from the current values at ``start_epoch`` to the specified ``final_coeffs`` over ``transition_epochs`` epochs.

    This callback is stateful and captures the loss coefficients at ``start_epoch`` for interpolation.

    .. note::
        This callback is currently in beta testing. Please report any unexpected behavior or issues.

    Example usage in config to transition to energy:force:stress = 1:1:1 over 200 epochs starting at epoch 100 (from whatever coefficients they were originally at):

    .. code-block:: yaml

        callbacks:
          - _target_: nequip.train.callbacks.LinearLossCoefficientScheduler
            final_coeffs:
              per_atom_energy_mse: 1.0
              forces_mse: 1.0
              stress_mse: 1.0
            start_epoch: 100
            transition_epochs: 200

    Multiple ``LinearLossCoefficientScheduler`` callbacks can be composed for multi-stage scheduling:

    .. code-block:: yaml

        callbacks:
          # First transition: current -> 1:5:1 from epoch 50-150
          - _target_: nequip.train.callbacks.LinearLossCoefficientScheduler
            final_coeffs:
              per_atom_energy_mse: 1.0
              forces_mse: 5.0
              stress_mse: 1.0
            start_epoch: 50
            transition_epochs: 100
          # Second transition: current -> 1:1:1 from epoch 200-400
          - _target_: nequip.train.callbacks.LinearLossCoefficientScheduler
            final_coeffs:
              per_atom_energy_mse: 1.0
              forces_mse: 1.0
              stress_mse: 1.0
            start_epoch: 200
            transition_epochs: 200

    .. warning::
        When composing multiple schedulers, ensure their epoch ranges do not overlap. No safety checks are performed to validate scheduler composition. Additionally, callback execution order is not guaranteed and training protocols should not rely on specific callback execution orders.

    Args:
        final_coeffs (Dict[str, float]): target loss coefficient dictionary
        start_epoch (int): epoch at which to start the transition (default: 0)
        transition_epochs (int): number of epochs over which to transition
    """

    def __init__(
        self,
        final_coeffs: Dict[str, float],
        transition_epochs: int,
        start_epoch: int = 0,
    ):
        # normalize final coefficients since captured coefficients will be normalized
        final_total = sum(final_coeffs.values())
        self.final_coeffs = {
            key: val / final_total for key, val in final_coeffs.items()
        }
        self.start_epoch = start_epoch
        self.transition_epochs = transition_epochs
        self.captured_initial_coeffs = None

        assert start_epoch >= 0, "Start epoch must be non-negative"
        assert transition_epochs > 0, "Transition epochs must be positive"

    def on_train_epoch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ) -> None:
        """"""
        current_epoch = trainer.current_epoch

        # NOTE: initial coeffs captured should already be normalized
        # final coeffs were normalized at __init__

        if current_epoch == self.start_epoch and self.captured_initial_coeffs is None:
            # lazily capture the current coefficients when we start
            self.captured_initial_coeffs = {
                metric_name: metric_dict["coeff"]
                for metric_name, metric_dict in pl_module.loss.metrics.items()
                if metric_name in self.final_coeffs
            }
            # sanity check that all `final_coeffs` keys are present in the metrics
            assert set(self.final_coeffs.keys()) == set(
                self.captured_initial_coeffs.keys()
            ), (
                f"Mismatch between `final_coeffs` keys {set(self.final_coeffs.keys())} and available metrics {set(self.captured_initial_coeffs.keys())}"
            )

        if (
            self.start_epoch
            < current_epoch
            <= self.start_epoch + self.transition_epochs
        ):
            # linear interpolation during transition period
            assert self.captured_initial_coeffs is not None, (
                "Initial coefficients should have been captured"
            )
            epochs_into_transition = current_epoch - self.start_epoch
            alpha = epochs_into_transition / self.transition_epochs
            interpolated_coeffs = {}

            for key in self.final_coeffs.keys():
                initial_val = self.captured_initial_coeffs[key]
                final_val = self.final_coeffs[key]
                interpolated_coeffs[key] = initial_val + alpha * (
                    final_val - initial_val
                )

            pl_module.loss.set_coeffs(interpolated_coeffs)

    def state_dict(self):
        """"""
        return {
            "final_coeffs": self.final_coeffs,
            "start_epoch": self.start_epoch,
            "transition_epochs": self.transition_epochs,
            "captured_initial_coeffs": self.captured_initial_coeffs,
        }

    def load_state_dict(self, state_dict):
        """"""
        self.final_coeffs = state_dict["final_coeffs"]
        self.start_epoch = state_dict["start_epoch"]
        self.transition_epochs = state_dict["transition_epochs"]
        self.captured_initial_coeffs = state_dict["captured_initial_coeffs"]
