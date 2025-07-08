import pytest
from nequip.utils.unittests.model_tests import BaseEnergyModelTests
from nequip.utils.versions import _TORCH_GE_2_4

try:
    import openequivariance  # noqa: F401

    _OEQ_INSTALLED = True
except ImportError:
    _OEQ_INSTALLED = False

BASIC_INFO = {
    "seed": 123,
    "type_names": ["H", "C", "O"],
    "r_max": 4.0,
    "avg_num_neighbors": 20,
    "per_type_energy_shifts": {"H": 3.45, "C": 5.67, "O": 7.89},
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


class TestNequIPModel(BaseEnergyModelTests):
    @pytest.fixture
    def strict_locality(self):
        return False

    @pytest.fixture(scope="class")
    def nequip_compile_tol(self, model_dtype):
        return {"float32": 5e-5, "float64": 1e-12}[model_dtype]

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

    @pytest.fixture(
        scope="class",
        params=[None]
        + (["enable_OpenEquivariance"] if _TORCH_GE_2_4 and _OEQ_INSTALLED else []),
    )
    def nequip_compile_acceleration_modifiers(self, request):
        """Test acceleration modifiers in nequip-compile workflows."""
        if request.param is None:
            return None

        def modifier_handler(mode, device, model_dtype):
            if request.param == "enable_OpenEquivariance":
                import openequivariance  # noqa: F401,F811

                # TODO: test when ready (likely PyTorch 2.8.0)
                if mode == "aotinductor":
                    pytest.skip("OEQ AOTI tests skipped for now")

                if device == "cpu":
                    pytest.skip("OEQ tests skipped for CPU")

                return ["enable_OpenEquivariance"]
            else:
                raise ValueError(f"Unknown modifier: {request.param}")

        return modifier_handler

    @pytest.fixture(
        scope="class",
        params=[None]
        + (["enable_OpenEquivariance"] if _TORCH_GE_2_4 and _OEQ_INSTALLED else []),
    )
    def train_time_compile_acceleration_modifiers(self, request):
        """Test acceleration modifiers in train-time compile workflows."""
        if request.param is None:
            return None

        def modifier_handler(device):
            if request.param == "enable_OpenEquivariance":
                import openequivariance  # noqa: F401,F811

                if device == "cpu":
                    pytest.skip("OEQ tests skipped for CPU")

                return [{"modifier": "enable_OpenEquivariance"}]
            else:
                raise ValueError(f"Unknown modifier: {request.param}")

        return modifier_handler
