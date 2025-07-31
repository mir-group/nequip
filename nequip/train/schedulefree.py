from .lightning import NequIPLightningModule
from typing import Dict, Any, Optional
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from nequip.utils import RankedLogger
import torch

logger = RankedLogger(__name__, rank_zero_only=True)


class ScheduleFreeLightningModule(NequIPLightningModule):
    """
    NequIP LightningModule using Facebook's Schedule-Free optimizer.

    This module wraps the model's optimizer in one of the Schedule-Free variants.
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

        self._schedulefree_state_dict: Optional[Dict[str, Any]] = None
        super().__init__(optimizer=optimizer, **kwargs)

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        sd = super().state_dict(*args, **kwargs)
        opt = self.optimizers()
        if opt is not None:
            try:
                opt.eval()
            except Exception as e:
                logger.warning(f"Schedule-Free optimizer eval() failed: {e}")
            sd["_schedulefree_optimizer_state"] = opt.state_dict()
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        sched_sd = state_dict.pop("_schedulefree_optimizer_state", None)
        super().load_state_dict(state_dict, strict=strict)
        if sched_sd is not None:
            logger.info("Loaded Schedule-Free optimizer state from checkpoint.")
            self._schedulefree_state_dict = sched_sd

    @property
    def evaluation_model(self) -> torch.nn.Module:
        opt = super().configure_optimizers()
        if self._schedulefree_state_dict:
            try:
                opt.load_state_dict(self._schedulefree_state_dict)
            except Exception as e:
                logger.warning(f"Failed to load Schedule-Free optimizer state: {e}")
        try:
            opt.eval()
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
