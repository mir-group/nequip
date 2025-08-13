from .lightning import NequIPLightningModule
from typing import Dict, Any
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from nequip.utils import RankedLogger
import torch

logger = RankedLogger(__name__, rank_zero_only=True)

# Note: Manual `.train()`/`.eval()` mode control for optimizer is required to ensure smoothed weights are captured at the right time
# Related discussion on Lightning timing hooks:
# https://github.com/Lightning-AI/pytorch-lightning/discussions/19759


class ScheduleFreeLightningModule(NequIPLightningModule):
    """
    NequIP LightningModule using Facebook's Schedule-Free optimizer.

    This module wraps the model's optimizer in one of Facebook's Schedule-Free variants.
    See: https://github.com/facebookresearch/schedule_free

    Args:
        optimizer (Dict[str, Any]): Dictionary that must include a _target_
            corresponding to one of the Schedule-Free optimizers and other keyword arguments
            compatible with the Schedule-Free variants.
    """

    def __init__(self, optimizer: Dict[str, Any], **kwargs):
        valid_targets = {
            "AdamWScheduleFree",
            "SGDScheduleFree",
            "RAdamScheduleFree",
        }
        target = optimizer.get("_target_")
        if not target or not any(target.endswith(name) for name in valid_targets):
            raise MisconfigurationException(
                f"Invalid optimizer: expected Schedule-Free optimizer (_target_ ending with one of {valid_targets}), "
                f"but found '{target}'"
            )

        # Will be used to lazily restore optimizer state in evaluation_model
        self._schedulefree_state_dict: Dict[str, Any] = {}

        super().__init__(optimizer=optimizer, **kwargs)

    #  Lightning Hook
    def on_save_checkpoint(self, checkpoint: dict):
        """"""
        # Schedule-Free optimizers require .eval() to expose smoothed weights.
        # This hook is called AFTER Lightning has already saved model/optimizer state,
        # so we only store the smoothed state_dict here for packaging.
        opt = self.optimizers()
        if opt is not None:
            checkpoint["schedulefree_optimizer_state_dict"] = opt.state_dict()

    #  Lightning Hook
    def on_load_checkpoint(self, checkpoint: dict):
        """"""
        # We extract our custom optimizer state for later lazy loading
        state = checkpoint.get("schedulefree_optimizer_state_dict")
        if state is not None:
            logger.info(
                "Storing Schedule-Free optimizer state from checkpoint for lazy loading."
            )
            self._schedulefree_state_dict = state

    #  NequIP-Specific Override for Packaging
    @property
    def evaluation_model(self) -> torch.nn.Module:
        # This is used during packaging to get the smoothed evaluation weights.
        logger.info("Loading Schedule-Free optimizer weights for evaluation.")

        prev_state_dict = getattr(self, "_schedulefree_state_dict", None)
        opt = self.configure_optimizers()

        if prev_state_dict:
            opt.load_state_dict(prev_state_dict)

        # Set optimizer to evaluation mode for smoothed weights
        opt.eval()

        return self.model

    #  Lightning Hook
    def on_train_epoch_start(self) -> None:
        """"""
        # Ensures fast weights are used during training
        self.optimizers().train()

    #  Lightning Hook
    def on_validation_epoch_start(self) -> None:
        """"""
        # Ensures smoothed weights are used for validation
        self.optimizers().eval()

    #  Lightning Hook
    def on_test_epoch_start(self) -> None:
        """"""
        # Ensures smoothed weights are used during testing
        self.optimizers().eval()

    #  Lightning Hook
    def on_predict_epoch_start(self) -> None:
        """"""
        # Ensures smoothed weights are used during prediction/inference
        self.optimizers().eval()
