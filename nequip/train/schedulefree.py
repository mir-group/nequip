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

        self.schedulefree_optimizer_class = optimizer["_target_"]
        super().__init__(optimizer=optimizer, **kwargs)

    @property
    def evaluation_model(self) -> torch.nn.Module:
        logger.info("Loading Schedule-Free optimizer weights for evaluation.")

        try:
            # Normal case when inside Trainer
            self.optimizers().eval()
        except Exception as e:
            logger.warning(f"Cannot call optimizer.eval(): {e}")
            # Trainerless fallback: Lightning stores restored optimizers here
            opt_list = getattr(
                getattr(self, "_optimizer_connector", None), "_optimizers", []
            )
            if not opt_list:
                logger.warning("No optimizer found for manual evaluation smoothing.")
            else:
                logger.info("Manually applying Schedule-Free z → param.data smoothing")
                for opt in opt_list:
                    for group in opt.param_groups:
                        beta1, _ = group.get("betas", (0.9, 0.999))
                        for p in group["params"]:
                            state = opt.state.get(p, {})
                            z = state.get("z")
                            if z is not None:
                                p.data.lerp_(z.to(p.device), 1 - 1 / beta1)

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
