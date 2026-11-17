# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import pytest
import torch

from nequip.model.modify_utils import modify
from nequip.nn.model_modifier_utils import model_modifier, replace_submodules


class Swappable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    @model_modifier(persistent=False)
    @classmethod
    def swap(cls, model):
        def factory(old):
            new = cls()
            new.load_state_dict(old.state_dict())
            return new

        return replace_submodules(model, cls, factory)


@pytest.mark.parametrize("training", [True, False])
def test_modify_keeps_training_mode(training):
    """Swapped-in modules match the model's training mode."""
    model = torch.nn.Sequential(Swappable()).train(training)
    old = model[0]
    modified = modify(model, [{"modifier": "swap"}])
    assert modified[0] is not old
    assert all(module.training == training for module in modified.modules())
