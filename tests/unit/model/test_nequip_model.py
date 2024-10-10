import pytest

from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.model import model_from_config
from nequip.nn import AtomwiseLinear
from nequip.utils.unittests.model_tests import BaseEnergyModelTests

COMMON_CONFIG = {
    "avg_num_neighbors": None,
    "type_names": ["H", "C", "O"],
    "seed": 123,
    # Just in case for when that builder exists:
    "pair_style": "ZBL",
    "ZBL_chemical_species": ["H", "C", "O"],
    "units": "metal",
}
r_max = 3
minimal_config1 = dict(
    irreps_edge_sh="0e + 1o",
    r_max=4,
    feature_irreps_hidden="4x0e + 4x1o",
    num_layers=2,
    num_basis=8,
    PolynomialCutoff_p=6,
    nonlinearity_type="norm",
    **COMMON_CONFIG,
)
minimal_config2 = dict(
    irreps_edge_sh="0e + 1o",
    r_max=4,
    chemical_embedding_irreps_out="8x0e + 8x0o + 8x1e + 8x1o",
    irreps_mid_output_block="2x0e",
    feature_irreps_hidden="4x0e + 4x1o",
    **COMMON_CONFIG,
)
minimal_config3 = dict(
    irreps_edge_sh="0e + 1o",
    r_max=4,
    feature_irreps_hidden="4x0e + 4x1o",
    num_layers=2,
    num_basis=8,
    PolynomialCutoff_p=6,
    nonlinearity_type="gate",
    **COMMON_CONFIG,
)
minimal_config4 = dict(
    irreps_edge_sh="0e + 1o + 2e",
    r_max=4,
    feature_irreps_hidden="2x0e + 2x1o + 2x2e",
    num_layers=2,
    num_basis=3,
    PolynomialCutoff_p=6,
    nonlinearity_type="gate",
    # test custom nonlinearities
    nonlinearity_scalars={"e": "silu", "o": "tanh"},
    nonlinearity_gates={"e": "silu", "o": "abs"},
    **COMMON_CONFIG,
)


class TestNequIPModel(BaseEnergyModelTests):
    @pytest.fixture
    def strict_locality(self):
        return False

    @pytest.fixture(
        params=[minimal_config1, minimal_config2, minimal_config3, minimal_config4],
        scope="class",
    )
    def base_config(self, request):
        return request.param

    @pytest.fixture(
        params=[
            (
                ["NequIPGNNEnergyModel", "ForceOutput"],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.FORCE_KEY,
                ],
            ),
            # # Save some time in the tests
            # (
            #     ["NequIPGNNEnergyModel"],
            #     [
            #         AtomicDataDict.TOTAL_ENERGY_KEY,
            #         AtomicDataDict.PER_ATOM_ENERGY_KEY,
            #     ],
            # ),
            (
                ["NequIPGNNEnergyModel", "StressForceOutput"],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.FORCE_KEY,
                    AtomicDataDict.STRESS_KEY,
                    AtomicDataDict.VIRIAL_KEY,
                ],
            ),
            (
                ["NequIPGNNEnergyModel", "PairPotentialTerm", "StressForceOutput"],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.FORCE_KEY,
                    AtomicDataDict.STRESS_KEY,
                    AtomicDataDict.VIRIAL_KEY,
                ],
            ),
        ],
        scope="class",
    )
    def config(self, request, base_config):
        config = base_config.copy()
        builder, out_fields = request.param
        config = config.copy()
        config["model_builders"] = builder
        return config, out_fields

    def test_submods(self):
        config = minimal_config2.copy()
        config["model_builders"] = ["NequIPGNNEnergyModel"]
        model = model_from_config(config=config, initialize=True)
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
