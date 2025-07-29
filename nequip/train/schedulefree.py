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
            opt = self.optimizers()
            opt.eval()
        except RuntimeError as e:
            logger.warning(f"Cannot call optimizer.eval(): {e}")
            # Attempt to manually recover optimizer from self.__dict__ (Lightning stores it here after load_from_checkpoint)
            for obj in self.__dict__.values():
                if isinstance(obj, torch.optim.Optimizer):
                    for group in obj.param_groups:
                        beta1, _ = group.get("betas", (0.9, 0.999))
                        for p in group["params"]:
                            state = obj.state.get(p, {})
                            z = state.get("z")
                            if z is not None:
                                p.data.lerp_(z.to(p.device), 1 - 1 / beta1)
                    logger.info(
                        "Manually applied z → param.data from Schedule-Free optimizer"
                    )
                    break
            else:
                logger.warning("No optimizer found for manual evaluation smoothing.")

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
