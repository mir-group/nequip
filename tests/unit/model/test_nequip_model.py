import pytest
from nequip.utils.unittests.model_tests_lammps import LAMMPSMLIAPIntegrationMixin
from nequip.utils.unittests.model_tests_torchsim import TorchSimIntegrationMixin
from nequip.utils.versions import _TORCH_GE_2_7, _TORCH_IS_2_10_0

try:
    import openequivariance  # noqa: F401

    _OEQ_INSTALLED = True
except ImportError:
    _OEQ_INSTALLED = False

try:
    import cuequivariance  # noqa: F401
    import cuequivariance_torch  # noqa: F401

    _CUEQ_INSTALLED = True
except ImportError:
    _CUEQ_INSTALLED = False

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
    num_features=[7, 13, 5],
    num_layers=2,
    # ZBL pair potential term
    pair_potential={
        "_target_": "nequip.nn.pair_potential.ZBL",
        "chemical_species": ["H", "C", "O"],
        "units": "metal",
    },
    **COMMON_CONFIG,
)


class TestNequIPModel(TorchSimIntegrationMixin, LAMMPSMLIAPIntegrationMixin):
    """NequIP model tests.

    Gets compilation tests via TorchSimIntegrationMixin → CompilationTestsMixin.
    """

    @pytest.fixture
    def strict_locality(self):
        return False

    @pytest.fixture(scope="class")
    def equivariance_tol(self, model_dtype):
        return {"float32": 5e-5, "float64": 1e-7}[model_dtype]

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
        + (["enable_OpenEquivariance"] if _TORCH_GE_2_7 and _OEQ_INSTALLED else [])
        + (["enable_CuEquivariance"] if _CUEQ_INSTALLED else []),
    )
    def nequip_compile_acceleration_modifiers(self, request):
        """Test acceleration modifiers in nequip-compile workflows."""
        if request.param is None:
            # for base NequIP models (no modifiers), skip CPU+aotinductor for PyTorch 2.10.0
            # due to known compilation bug; left for future PyTorch versions to resolve
            def modifier_handler(mode, device, model_dtype):
                if device == "cpu" and mode == "aotinductor" and _TORCH_IS_2_10_0:
                    pytest.skip(
                        "CPU + aotinductor compilation is known to fail for NequIP models in PyTorch 2.10.0"
                    )
                return None

            return modifier_handler

        def modifier_handler(mode, device, model_dtype):
            if request.param == "enable_OpenEquivariance":
                import openequivariance  # noqa: F401,F811
                from nequip.utils.versions import _TORCH_GE_2_9

                # OEQ + AOTI requires PyTorch >= 2.9
                if mode == "aotinductor" and not _TORCH_GE_2_9:
                    pytest.skip("OEQ AOTI requires PyTorch >= 2.9")

                if device == "cpu":
                    pytest.skip("OEQ tests skipped for CPU")

                return ["enable_OpenEquivariance"]
            elif request.param == "enable_CuEquivariance":
                import cuequivariance  # noqa: F401,F811
                import cuequivariance_torch  # noqa: F401,F811

                if model_dtype == "float64":
                    pytest.skip("CuEq tests skipped for f64 models")

                if device == "cpu":
                    pytest.skip("CuEq tests skipped for CPU")

                return ["enable_CuEquivariance"]
            else:
                raise ValueError(f"Unknown modifier: {request.param}")

        return modifier_handler

    @pytest.fixture(
        scope="class",
        params=[None]
        + (["enable_OpenEquivariance"] if _TORCH_GE_2_7 and _OEQ_INSTALLED else []),
        # + (["enable_CuEquivariance"] if _CUEQ_INSTALLED else []),
        # NOTE: ^ some tests fail with CuEq
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
            elif request.param == "enable_CuEquivariance":
                import cuequivariance  # noqa: F401,F811
                import cuequivariance_torch  # noqa: F401,F811

                if device == "cpu":
                    pytest.skip("CuEq tests skipped for CPU")

                return [{"modifier": "enable_CuEquivariance"}]
            else:
                raise ValueError(f"Unknown modifier: {request.param}")

        return modifier_handler

    @pytest.fixture(
        scope="class",
        params=[None]
        + (["enable_OpenEquivariance"] if _TORCH_GE_2_7 and _OEQ_INSTALLED else [])
        + (["enable_CuEquivariance"] if _CUEQ_INSTALLED else []),
    )
    def mliap_acceleration_modifiers(self, request):
        """Test acceleration modifiers in MLIAP workflows."""

        def modifier_handler(compile, model_dtype):
            # skip float64 for NequIP models as noted in existing integration tests
            if model_dtype == "float64":
                pytest.skip("Skipping f64 ML-IAP tests for NequIP.")

            if request.param is None:
                return []
            elif request.param == "enable_OpenEquivariance":
                import openequivariance  # noqa: F401,F811

                return ["enable_OpenEquivariance"]
            elif request.param == "enable_CuEquivariance":
                import cuequivariance  # noqa: F401,F811
                import cuequivariance_torch  # noqa: F401,F811

                if model_dtype == "float64":
                    pytest.skip("CuEq tests skipped for f64 models")

                return ["enable_CuEquivariance"]
            else:
                raise ValueError(f"Unknown modifier: {request.param}")

        return modifier_handler
