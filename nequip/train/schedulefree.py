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

        self._schedulefree_state_dict: Dict[str, Any] = {}
        super().__init__(optimizer=optimizer, **kwargs)

    def on_save_checkpoint(self, checkpoint: dict):
        opt = self.optimizers()
        if opt is not None:
            logger.info("Applying smoothing before saving checkpoint...")
            train_modes = [group.get("train_mode", False) for group in opt.param_groups]
            for group in opt.param_groups:
                group["train_mode"] = True
            with torch.no_grad():
                opt.eval()
            # Overwrite optimizer and model state in checkpoint
            checkpoint["optimizer_states"] = [opt.state_dict()]
            checkpoint["state_dict"] = self.state_dict()
            # Restore train_mode flags
            for group, mode in zip(opt.param_groups, train_modes):
                group["train_mode"] = mode
            checkpoint["schedulefree_optimizer_state_dict"] = opt.state_dict()

    def on_load_checkpoint(self, checkpoint: dict):
        state = checkpoint.get("schedulefree_optimizer_state_dict")
        if state is not None:
            logger.info(
                "Storing Schedule-Free optimizer state from checkpoint for lazy loading."
            )
            self._schedulefree_state_dict = state

    @property
    def evaluation_model(self) -> torch.nn.Module:
        logger.info("Loading Schedule-Free optimizer weights for evaluation.")
        opt = super().configure_optimizers()
        if self._schedulefree_state_dict:
            try:
                opt.load_state_dict(self._schedulefree_state_dict)
                logger.info("Loaded Schedule-Free optimizer state.")
            except Exception as e:
                logger.warning(f"Failed to load Schedule-Free optimizer state: {e}")
        try:
            with torch.no_grad():
                opt.eval()
        except Exception as e:
            logger.warning(f"Schedule-Free optimizer eval() failed: {e}")
        self.model.eval()
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
