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
def unit_orthogonal_init_(t: torch.Tensor):
    """Orthogonal init with <x_i^2> = 1"""
    assert t.ndim == 2
    torch.nn.init.orthogonal_(t, gain=math.sqrt(max(t.shape)))


# TODO: more inits


def _xavier_initialize_fcs(mod: torch.nn.Module):
    """Initialize ``e3nn.nn.FullyConnectedNet``s and ``torch.nn.Linear``s with Xavier uniform initialization"""
    if isinstance(mod, e3nn.nn.FullyConnectedNet):
        for layer in mod:
            # in FC:
            # h_in, _h_out = W.shape
            # W = W / h_in**0.5
            torch.nn.init.xavier_uniform_(
                layer.weight, gain=layer.weight.shape[0] ** 0.5
            )
    elif isinstance(mod, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(mod.weight)


def xavier_initialize_FCs(model: GraphModuleMixin, initialize: bool):
    if initialize:
        with torch.no_grad():
            model.apply(_xavier_initialize_fcs)
    return model
