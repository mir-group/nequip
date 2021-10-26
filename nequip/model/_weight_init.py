import math

import torch

import e3nn.o3
import e3nn.nn

from nequip.nn import GraphModuleMixin
from nequip.utils import Config


# == Load old state ==
def initialize_from_state(config: Config, model: GraphModuleMixin, initialize: bool):
    """Initialize the model from the state dict file given by the config options `initial_model_state`."""
    if not initialize:
        return model  # do nothing
    key = "initial_model_state"
    if key not in config:
        raise KeyError(
            f"initialize_from_state requires the `{key}` option specifying the state to initialize from"
        )
    state = torch.load(config[key])
    model.load_state_dict(state)
    return model


# == Init functions ==
def unit_uniform_init_(t: torch.Tensor):
    """Uniform initialization with <x_i^2> = 1"""
    t.uniform_(-math.sqrt(3), math.sqrt(3))


# TODO: does this normalization make any sense
# def unit_orthogonal_init_(t: torch.Tensor):
#     """Orthogonal init with <x_i^2> = 1"""
#     assert t.ndim == 2
#     torch.nn.init.orthogonal_(t, gain=math.sqrt(max(t.shape)))


def uniform_initialize_FCs(model: GraphModuleMixin, initialize: bool):
    """Initialize ``e3nn.nn.FullyConnectedNet``s with unit uniform initialization"""
    if initialize:

        def _uniform_initialize_fcs(mod: torch.nn.Module):
            if isinstance(mod, e3nn.nn.FullyConnectedNet):
                for layer in mod:
                    # in FC, normalization is expected
                    unit_uniform_init_(layer.weight)

        with torch.no_grad():
            model.apply(_uniform_initialize_fcs)
    return model
