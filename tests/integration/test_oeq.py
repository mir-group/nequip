from conftest import TrainingInvarianceBaseTest
import pytest
import torch
from nequip.utils.versions import _TORCH_GE_2_4


@pytest.mark.skipif(not _TORCH_GE_2_4, reason="OpenEquivariance requires torch >= 2.4")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="OEQ requires CUDA")
class TestOEQTrainingInvariance(TrainingInvarianceBaseTest):
    # only test with EMALightningModule for OEQ
    _TRAINING_MODULES_TO_TEST = ["nequip.train.EMALightningModule"]

    def modify_model_config(self, original_config):
        try:
            import openequivariance  # noqa: F401
        except ImportError:
            pytest.skip("OpenEquivariance not installed")
        new_config = original_config.copy()
        training_module = new_config["training_module"]
        original_model = training_module["model"]
        training_module["model"] = {
            "_target_": "nequip.model.modify",
            "modifiers": [{"modifier": "enable_OpenEquivariance"}],
            "model": original_model,
        }
        return new_config

    def map_location(self):
        return "cuda"
