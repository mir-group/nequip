import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.nn.grad_output import ForceStressOutput
from nequip.nn._graph_mixin import GraphModuleMixin
from nequip.nn import GraphModel
from nequip.model.modify_utils import modify


class SimpleEnergyModel(GraphModuleMixin, torch.nn.Module):
    """Simple energy model that depends on positions."""

    def __init__(self):
        super().__init__()
        # simple energy function: sum of squared distances from origin
        self._init_irreps(
            irreps_in={},
            irreps_out={AtomicDataDict.TOTAL_ENERGY_KEY: "0e"},
        )

    def forward(self, data):
        pos = data[AtomicDataDict.POSITIONS_KEY]
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = pos.square().sum().sqrt().view(1, 1)
        return data


@pytest.fixture
def energy_model():
    """Create a simple energy model."""
    return SimpleEnergyModel()


@pytest.fixture
def dummy_batch():
    """Create minimal test batch with positions."""
    return {
        AtomicDataDict.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], requires_grad=True
        ),
    }


def test_force_stress_modifiers(energy_model, dummy_batch):
    """Test enable/disable ForceStressOutput modifiers."""
    force_stress_module = ForceStressOutput(func=energy_model, do_derivatives=True)
    model = GraphModel(force_stress_module)

    # test initial state - forces should be present
    result_with_forces = model(dummy_batch.copy())
    assert AtomicDataDict.FORCE_KEY in result_with_forces

    # apply disable modifier
    disabled_model = modify(model, [{"modifier": "disable_ForceStressOutput"}])

    # test disabled state - forces should be absent
    result_no_forces = disabled_model(dummy_batch.copy())
    assert AtomicDataDict.FORCE_KEY not in result_no_forces
    assert (
        AtomicDataDict.TOTAL_ENERGY_KEY in result_no_forces
    )  # energy should still be there

    # apply enable modifier
    enable_args = {"modifier": "enable_ForceStressOutput"}
    enabled_model = modify(disabled_model, [enable_args])

    # test enabled state - forces should be present again
    result_forces_restored = enabled_model(dummy_batch.copy())
    assert AtomicDataDict.FORCE_KEY in result_forces_restored
