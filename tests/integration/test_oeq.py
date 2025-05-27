from nequip.utils.unittests.model_tests import BaseModelTests
from conftest import TrainingInvarianceBaseTest 
import pytest
import torch

from nequip.utils.test import override_irreps_debug

# =================== Unit tests ==========================

# Copied from unit test file
BASIC_INFO = {
    "seed": 123,
    "type_names": ["H", "C", "O"],
    "r_max": 4.0,
    "avg_num_neighbors": 3.0,
}

COMMON_CONFIG = {
    "_target_": "nequip.model.NequIPGNNModel",
    "l_max": 2,
    "parity": True,
    "radial_mlp_depth": 1,
    "radial_mlp_width": 8,
    **BASIC_INFO,
}

COMMON_FULL_CONFIG = {
    "_target_": "nequip.model.FullNequIPGNNModel",
    "radial_mlp_depth": [1, 2],
    "radial_mlp_width": [5, 7],
    **BASIC_INFO,
}

minimal_config1 = dict(
    num_features=8,
    num_layers=2,
    # ZBL pair potential term
    pair_potential={
        "_target_": "nequip.nn.pair_potential.ZBL",
        "chemical_species": ["H", "C", "O"],
        "units": "metal",
    },
    **COMMON_CONFIG,
)
minimal_config2 = dict(
    num_features=8,
    num_layers=3,
    per_type_energy_shifts=[3.45, 5.67, 7.89],
    **COMMON_CONFIG,
)
minimal_config3 = dict(
    num_features=8,
    num_layers=2,
    per_edge_type_cutoff={"H": 2.0, "C": {"H": 4.0, "C": 3.5, "O": 3.7}, "O": 3.9},
    **COMMON_CONFIG,
)
minimal_config4 = dict(
    irreps_edge_sh="0e + 1o",
    type_embed_num_features=11,
    feature_irreps_hidden=["13x0e + 4x1o", "7x0e"],
    convnet_nonlinearity_type="norm",
    # ZBL pair potential term
    pair_potential={
        "_target_": "nequip.nn.pair_potential.ZBL",
        "chemical_species": ["H", "C", "O"],
        "units": "metal",
    },
    **COMMON_FULL_CONFIG,
)

class TestOEQUnit(BaseModelTests):
    @pytest.fixture
    def strict_locality(self):
        return False

    @pytest.fixture(
        params=[
            minimal_config1,
            minimal_config2,
            minimal_config3,
            minimal_config4,
        ],
        scope="class",
    )
    def config(self, request):
        config = request.param
        config = config.copy()
        return config

    @pytest.mark.skip
    def test_init():
        pass

    @pytest.mark.skip
    def test_jit():
        pass

    @pytest.mark.skip
    def test_compile():
        pass

    @pytest.mark.skip
    def test_aot_export():
        pass

    @pytest.mark.skip
    def test_forward():
        pass

    @pytest.mark.skip
    def test_wrapped_unwrapped():
        pass

    @pytest.mark.skip
    def test_batch():
        pass

    @pytest.mark.skip
    def test_equivariance():
        pass

    @pytest.mark.skip
    def test_embedding_cutoff():
        pass

    @override_irreps_debug(False)
    def test_oeq(self, model, model_test_data, device):
        if device == "cpu":
            pytest.skip(
                "OEQ tests skipped for CPU" 
            )

        instance, config, _ = model
        # get tolerance based on model_dtype
        tol = {
            torch.float32: 5e-5,
            torch.float64: 1e-12,
        }[instance.model_dtype]

        # Make OEQ model
        config = {
            "_target_": "nequip.model.modify",
            "modifiers": [{"modifier": "enable_OpenEquivariance"}],
            "model": config.copy()
        } 
        oeq_model = self.make_model(config, device=device)

        self.compare_output_and_gradients(
            modelA=instance,
            modelB=oeq_model,
            model_test_data=model_test_data,
            tol=tol,
            compare_outputs=False # Internal checks guarantee that outputs are the same 
        )
    

# =================== Integration tests ==========================

class TestOEQTrainingInvariance(TrainingInvarianceBaseTest):
    def modify_model_config(self, original_config):
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