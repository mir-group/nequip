# This file is a part of the `nequip` package.
# Please see LICENSE and README at the root for information on using it.
import lightning
from lightning.pytorch.callbacks import Callback
from nequip.train import NequIPLightningModule
from nequip.utils.global_state import (
    set_global_state,
    get_latest_global_state,
    TF32_KEY,
)
from typing import Dict, Optional


class TF32Scheduler(Callback):
    """Schedule TF32 precision during training.

    The ``TF32Scheduler`` takes a single argument ``schedule``, which is a
    ``Dict[int, bool]`` where the keys are the epochs at which TF32 changes
    and the values are:

    - ``True``: Enable TF32 (faster but less precise)
    - ``False``: Disable TF32 (slower but more precise)

    Example usage in config:

    .. code-block:: yaml

        callbacks:
          - _target_: nequip.train.callbacks.TF32Scheduler
            schedule:
              0: true      # Start with TF32 enabled
              100: false   # Disable TF32 at epoch 100
              200: true    # Re-enable TF32 at epoch 200

    .. note::
        The schedule must start at epoch 0, and the initial setting must match
        your ``global_options.allow_tf32`` configuration.

    .. note::
        This callback is currently in beta testing. Please report any unexpected behavior or issues.


    Args:
        schedule (Dict[int, bool]): map of epoch to TF32 enabled/disabled
    """

    def __init__(self, schedule: Dict[int, bool]):
        # ensure that the keys are `int`s
        self.schedule = {int(k): v for k, v in schedule.items()}
        # sanity check - epochs are >= 0
        assert all([epoch >= 0 for epoch in self.schedule.keys()])
        assert 0 in self.schedule, "First epoch in TF32 scheduler must be 0"
        # NOTE: callback is instantiated only during training context,
        # and after global state has been loaded,
        # so this check should always work
        assert self.schedule[0] == get_latest_global_state()[TF32_KEY], (
            "Initial TF32Scheduler setting (epoch 0) must match global state "
            "(found under global_options in yaml config)"
        )

        # Initialize state for restarts
        self.last_tf32_setting = self.schedule[0]

    def on_train_epoch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ) -> None:
        """"""
        if trainer.current_epoch in self.schedule:
            self._set_tf32(self.schedule[trainer.current_epoch], pl_module)

    def _set_tf32(
        self, enabled: bool, pl_module: Optional[NequIPLightningModule] = None
    ):
        set_global_state(allow_tf32=enabled)
        self.last_tf32_setting = enabled
        if pl_module is not None:
            pl_module.log(
                "tf32_enabled",
                float(enabled),
                on_step=False,
                on_epoch=True,
            )

    def state_dict(self) -> Dict:
        """"""
        return {
            "last_tf32_setting": self.last_tf32_setting,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """"""
        # restore the last TF32 state from checkpoint
        self.last_tf32_setting = state_dict["last_tf32_setting"]
        self._set_tf32(self.last_tf32_setting)
