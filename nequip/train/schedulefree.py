from .lightning import NequIPLightningModule
from typing import Dict, Any
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from nequip.utils import RankedLogger
import torch

logger = RankedLogger(__name__, rank_zero_only=True)


class ScheduleFreeLightningModule(NequIPLightningModule):
    """
    NequIP LightningModule using Facebook's Schedule-Free optimizer.

    This module wraps the model's optimizer in one of Facebook's Schedule-Free variants.
    See: https://github.com/facebookresearch/schedule_free

    Args:
        optimizer (Dict[str, Any]): Dictionary that must include a `_target_`
            corresponding to one of the Schedule-Free optimizers and other keyword arguments
            compatible with the Schedule-Free variants.
    """

    def __init__(self, optimizer: Dict[str, Any], **kwargs):
        valid_targets = {
            "AdamWScheduleFree",
            "SGDScheduleFree",
            "RAdamScheduleFree",
        }
        if "_target_" not in optimizer or not any(
            optimizer["_target_"].endswith(name) for name in valid_targets
        ):
            raise MisconfigurationException(
                f"Invalid optimizer: expected Schedule-Free optimizer (_target_ ending with one of {valid_targets}), "
                f"but found '{optimizer['_target_']}'"
            )

        self._optimizer_config = optimizer
        self._schedulefree_optimizer = None
        super().__init__(optimizer=optimizer, **kwargs)

    def configure_optimizers(self):
        optim = super().configure_optimizers()
        self._schedulefree_optimizer = optim
        return optim

    def on_save_checkpoint(self, checkpoint: dict):
        if self._schedulefree_optimizer is not None:
            checkpoint["schedulefree_optimizer_state_dict"] = (
                self._schedulefree_optimizer.state_dict()
            )

    def on_load_checkpoint(self, checkpoint: dict):
        # Don't instantiate anything here — only store the state for later
        if "schedulefree_optimizer_state_dict" in checkpoint:
            logger.info(
                "Schedule-Free optimizer state found in checkpoint for evaluation."
            )
            self._schedulefree_optimizer_state_to_restore = checkpoint[
                "schedulefree_optimizer_state_dict"
            ]

    @property
    def evaluation_model(self) -> torch.nn.Module:
        logger.info("Preparing Schedule-Free evaluation model.")

        if self._schedulefree_optimizer is None:
            logger.debug("Instantiating Schedule-Free optimizer for evaluation.")
            try:
                self._schedulefree_optimizer = self.configure_optimizers()
            except Exception as e:
                logger.warning(f"Failed to instantiate Schedule-Free optimizer: {e}")
                return self.model

        if hasattr(self, "_schedulefree_optimizer_state_to_restore"):
            try:
                self._schedulefree_optimizer.load_state_dict(
                    self._schedulefree_optimizer_state_to_restore
                )
                logger.debug("Schedule-Free optimizer state successfully restored.")
            except Exception as e:
                logger.warning(f"Failed to load Schedule-Free optimizer state: {e}")
            del self._schedulefree_optimizer_state_to_restore  # Clean up after loading

        try:
            self._schedulefree_optimizer.eval()
        except Exception as e:
            logger.warning(f"Schedule-Free optimizer eval() failed: {e}")

        return self.model

    def on_fit_start(self) -> None:
        self.optimizers().train()

    def on_validation_model_eval(self) -> None:
        self.model.eval()
        self.optimizers().eval()

    def on_validation_model_train(self) -> None:
        self.model.train()
        self.optimizers().train()

    def on_test_model_eval(self) -> None:
        self.model.eval()
        self.optimizers().eval()

    def on_test_model_train(self) -> None:
        self.model.train()
        self.optimizers().train()

    def on_predict_model_eval(self) -> None:
        self.model.eval()
        self.optimizers().eval()
