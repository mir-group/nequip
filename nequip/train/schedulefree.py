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
            corresponding to one of the Schedule‑Free optimizers and other keyword arguments
            compatible with the Schedule‑Free variants.
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
                f"Invalid optimizer: expected Schedule‑Free optimizer (_target_ ending with one of {valid_targets}), "
                f"but found '{optimizer.get('_target_')}'"
            )

        self._optimizer_config = optimizer
        super().__init__(optimizer=optimizer, **kwargs)

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str, map_location=None, **kwargs: Any
    ) -> "ScheduleFreeLightningModule":
        ckpt = torch.load(checkpoint_path, map_location=map_location or "cpu")
        module: ScheduleFreeLightningModule = super().load_from_checkpoint(
            checkpoint_path, map_location=map_location, **kwargs
        )
        module.on_load_checkpoint(ckpt)
        return module

    def on_fit_start(self) -> None:
        optim = self.optimizers()
        self._schedulefree_optimizer = optim
        optim.train()

    def on_save_checkpoint(self, checkpoint: dict):
        opt = getattr(self, "_schedulefree_optimizer", None)
        if opt is None:
            opt = super().configure_optimizers()
            self._schedulefree_optimizer = opt
        checkpoint["schedulefree_optimizer_state_dict"] = opt.state_dict()

    def on_load_checkpoint(self, checkpoint: dict):
        state = checkpoint.get("schedulefree_optimizer_state_dict", None)
        if state is not None:
            logger.info("Restoring Schedule‑Free optimizer state from checkpoint.")
            if not hasattr(self, "_schedulefree_optimizer"):
                # Build a fresh optimizer via the base class
                self._schedulefree_optimizer = super().configure_optimizers()
            self._schedulefree_optimizer.load_state_dict(state)

    @property
    def evaluation_model(self) -> torch.nn.Module:
        logger.info("Applying Schedule‑Free smoothing for evaluation.")
        opt = getattr(self, "_schedulefree_optimizer", None)
        if opt is not None:
            try:
                opt.eval()
            except Exception as e:
                logger.warning(f"Schedule‑Free optimizer eval() failed: {e}")
        else:
            logger.warning("No Schedule‑Free optimizer found—skipping smoothing.")
        return self.model

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
