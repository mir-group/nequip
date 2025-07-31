from typing import Dict, Any

import torch
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from nequip.utils import RankedLogger

from .lightning import NequIPLightningModule

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

    _VALID_TARGETS = {"AdamWScheduleFree", "SGDScheduleFree", "RAdamScheduleFree"}

    def __init__(self, optimizer: Dict[str, Any], **kwargs):
        target = optimizer.get("_target_")
        if not target or not any(target.endswith(t) for t in self._VALID_TARGETS):
            raise MisconfigurationException(
                f"ScheduleFreeLightningModule got invalid optimiser target "
                f"'{target}'. Expected one of {sorted(self._VALID_TARGETS)}."
            )
        self._schedulefree_state_dict: Dict[str, Any] = {}
        self._inference_opt = None
        self._sf_eval_active = False

        super().__init__(optimizer=optimizer, **kwargs)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        opt = self.optimizers()
        if opt is not None:
            checkpoint["schedulefree_optimizer_state_dict"] = opt.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state = checkpoint.get("schedulefree_optimizer_state_dict")
        if not state:
            return

        self._schedulefree_state_dict = state
        if getattr(self, "_trainer", None) is None:
            logger.info("Applying Schedule‑Free evaluation weights on load.")
            opt = super().configure_optimizers()
            try:
                opt.load_state_dict(state)
                opt.eval()
            except Exception as err:
                logger.warning(f"SF weight swap failed: {err!r}")
            self._inference_opt = opt

    @property
    def evaluation_model(self) -> torch.nn.Module:
        if getattr(self, "_trainer", None) is not None and not self._sf_eval_active:
            try:
                self.optimizers().eval()
            except Exception as err:
                logger.warning(f"SF optimiser eval() failed: {err!r}")
            self._sf_eval_active = True
        return self.model

    def _reset_sf_flag(self) -> None:
        self._sf_eval_active = False

    def on_fit_start(self) -> None:
        self.optimizers().train()

    def on_validation_model_eval(self) -> None:
        self.optimizers().eval()

    def on_validation_model_train(self) -> None:
        self.optimizers().train()
        self._reset_sf_flag()

    def on_test_model_eval(self) -> None:
        self.optimizers().eval()

    def on_test_model_train(self) -> None:
        self.optimizers().train()
        self._reset_sf_flag()

    def on_predict_model_eval(self) -> None:
        self.optimizers().eval()
