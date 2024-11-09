import pytest

from e3nn import o3

from nequip.nn import AtomwiseLinear
from nequip.utils.unittests.model_tests import BaseEnergyModelTests

from hydra.utils import instantiate


BASIC_INFO = {
    "seed": 123,
    "type_names": ["H", "C", "O"],
    "r_max": 4.0,
}

COMMON_CONFIG = {
    "_target_": "nequip.model.NequIPGNNModel",
    "l_max": 1,
    "parity": True,
    "invariant_layers": 1,
    "invariant_neurons": 8,
    **BASIC_INFO,
}

COMMON_FULL_CONFIG = {
    "_target_": "nequip.model.FullNequIPGNNModel",
    "invariant_layers": [1, 2],
    "invariant_neurons": [5, 7],
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
    num_features=4,
    num_layers=3,
    per_type_energy_shifts=[3.45, 5.67, 7.89],
    **COMMON_CONFIG,
)
minimal_config3 = dict(
    num_features=4,
    num_layers=2,
    per_edge_type_cutoff={"H": 2.0, "C": {"H": 4.0, "C": 3.5, "O": 3.7}, "O": 3.9},
    **COMMON_CONFIG,
)
minimal_config4 = dict(
    irreps_edge_sh="0e + 1o",
    chemical_embedding_irreps_out="4x0e + 11x1o",
    feature_irreps_hidden=["13x0e + 4x1o", "3x0e + 5x1o"],
    conv_to_output_hidden_irreps_out="3x0e + 7x1o",
    convnet_nonlinearity_type="norm",
    # ZBL pair potential term
    pair_potential={
        "_target_": "nequip.nn.pair_potential.ZBL",
        "chemical_species": ["H", "C", "O"],
        "units": "metal",
    },
    **COMMON_FULL_CONFIG,
)
minimal_config5 = dict(
    irreps_edge_sh="0e + 1o + 2e",
    chemical_embedding_irreps_out="7x0e + 3x1o",
    feature_irreps_hidden=["2x0e + 2x1o + 2x2e"] * 2,
    conv_to_output_hidden_irreps_out="5x0e + 2x1o",
    num_bessels=12,
    # test custom nonlinearities
    convnet_nonlinearity_gates={"e": "silu", "o": "abs"},
    per_type_energy_shifts=[3.45, 5.67, 7.89],
    **COMMON_FULL_CONFIG,
)


class TestNequIPModel(BaseEnergyModelTests):
    @pytest.fixture
    def strict_locality(self):
        return False

    @pytest.fixture(
        params=[minimal_config1, minimal_config2, minimal_config3, minimal_config4],
        scope="class",
    )
    def config(self, request):
        config = request.param
        config = config.copy()
        return config

    @pytest.mark.skip("ignore for now")
    def test_submods(self):
        config = minimal_config2.copy()
        model = instantiate(config, _recursive_=False)
        chemical_embedding = model.model.chemical_embedding
        assert isinstance(chemical_embedding, AtomwiseLinear)
        true_irreps = o3.Irreps(minimal_config2["chemical_embedding_irreps_out"])
        assert (
            chemical_embedding.irreps_out[chemical_embedding.out_field] == true_irreps
        )
        # Make sure it propagates
        assert (
            model.model.layer0_convnet.irreps_in[chemical_embedding.out_field]
            == true_irreps
        )
