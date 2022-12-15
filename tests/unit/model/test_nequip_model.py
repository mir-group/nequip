import pytest
import torch

from e3nn import o3

from nequip.data import AtomicDataDict, AtomicData
from nequip.model import model_from_config
from nequip.nn import AtomwiseLinear
from nequip.utils.unittests.model_tests import BaseEnergyModelTests

COMMON_CONFIG = {
    "avg_num_neighbors": None,
    "num_types": 3,
    "types_names": ["H", "C", "O"],
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
    **COMMON_CONFIG
)
minimal_config2 = dict(
    irreps_edge_sh="0e + 1o",
    r_max=4,
    chemical_embedding_irreps_out="8x0e + 8x0o + 8x1e + 8x1o",
    irreps_mid_output_block="2x0e",
    feature_irreps_hidden="4x0e + 4x1o",
    **COMMON_CONFIG
)
minimal_config3 = dict(
    irreps_edge_sh="0e + 1o",
    r_max=4,
    feature_irreps_hidden="4x0e + 4x1o",
    num_layers=2,
    num_basis=8,
    PolynomialCutoff_p=6,
    nonlinearity_type="gate",
    **COMMON_CONFIG
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
    **COMMON_CONFIG
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
                ["EnergyModel", "ForceOutput"],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.FORCE_KEY,
                ],
            ),
            (
                ["EnergyModel"],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                ],
            ),
            (
                ["EnergyModel", "StressForceOutput"],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.FORCE_KEY,
                    AtomicDataDict.STRESS_KEY,
                    AtomicDataDict.VIRIAL_KEY,
                ],
            ),
            (
                ["EnergyModel", "ParaStressForceOutput"],
                [
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    AtomicDataDict.FORCE_KEY,
                    AtomicDataDict.STRESS_KEY,
                    AtomicDataDict.VIRIAL_KEY,
                    AtomicDataDict.ATOM_VIRIAL_KEY,
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
        config["model_builders"] = ["EnergyModel"]
        model = model_from_config(config=config, initialize=True)
        assert isinstance(model.chemical_embedding, AtomwiseLinear)
        true_irreps = o3.Irreps(minimal_config2["chemical_embedding_irreps_out"])
        assert (
            model.chemical_embedding.irreps_out[model.chemical_embedding.out_field]
            == true_irreps
        )
        # Make sure it propagates
        assert (
            model.layer0_convnet.irreps_in[model.chemical_embedding.out_field]
            == true_irreps
        )

    def test_stress(self, atomic_bulk_batch, device):
        config_og = minimal_config2.copy()
        config_og["model_builders"] = ["EnergyModel", "StressForceOutput"]
        model_og = model_from_config(config=config_og, initialize=True)
        nn_state = model_og.state_dict()
        
        config_para = minimal_config2.copy()
        config_para["model_builders"] = ["EnergyModel", "ParaStressForceOutput"]
        model_para = model_from_config(config=config_para, initialize=True)
        
        model_para.load_state_dict(nn_state, strict = True)
        
        model_og.to(device)
        model_para.to(device)
        data = atomic_bulk_batch.to(device)
        
        output_og = model_og(AtomicData.to_AtomicDataDict(data))
        output_para = model_para(AtomicData.to_AtomicDataDict(data))
        
        assert torch.allclose(output_og[AtomicDataDict.STRESS_KEY], output_para[AtomicDataDict.STRESS_KEY], atol=1e-6)
        assert torch.allclose(output_og[AtomicDataDict.VIRIAL_KEY], output_para[AtomicDataDict.VIRIAL_KEY], atol=1e-5) # Little big here caused by the summation over edges.

