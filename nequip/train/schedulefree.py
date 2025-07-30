from .lightning import NequIPLightningModule
from typing import Dict, Any
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from nequip.utils import RankedLogger
from hydra.utils import instantiate
import torch

logger = RankedLogger(__name__, rank_zero_only=True)


class ScheduleFreeLightningModule(NequIPLightningModule):
    """
    NequIP LightningModule using Facebook's Schedule-Free optimizer.

    This module wraps the model's optimizer in one of Facebook's Schedule-Free variants.
    See: https://github.com/facebookresearch/schedule_free

    Args:
        optimizer (Dict[str, Any]): A Hydra config dict with `_target_` pointing to a Schedule-Free optimizer.
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
        self._schedulefree_state_dict = None
        super().__init__(optimizer=optimizer, **kwargs)

    def on_save_checkpoint(self, checkpoint: dict):
        try:
            opt = self.optimizers()
            checkpoint["schedulefree_optimizer_state_dict"] = opt.state_dict()
            logger.info("Saved Schedule-Free optimizer state for evaluation model.")
        except Exception as e:
            logger.warning(f"Could not save schedule-free optimizer state: {e}")

    def on_load_checkpoint(self, checkpoint: dict):
        self._schedulefree_state_dict = checkpoint.get(
            "schedulefree_optimizer_state_dict", None
        )

    @property
    def evaluation_model(self) -> torch.nn.Module:
        logger.info("Loading Schedule-Free optimizer weights for evaluation.")
        try:
            opt = instantiate(self._optimizer_config, params=self.model.parameters())

            if self._schedulefree_state_dict is not None:
                opt.load_state_dict(self._schedulefree_state_dict)
                opt.eval()
                logger.info("Schedule-Free optimizer eval() applied for smoothing.")
            else:
                logger.warning("No optimizer state found — skipping smoothing.")
        except Exception as e:
            logger.warning(f"Schedule-Free optimizer smoothing failed: {e}")

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

    def on_validation_epoch_end(self) -> None:
        try:
            self.optimizers().eval()
            logger.info(
                "Schedule-Free optimizer set to eval mode before checkpointing."
            )
        except Exception as e:
            logger.warning(f"Could not set optimizer to eval() before checkpoint: {e}")
