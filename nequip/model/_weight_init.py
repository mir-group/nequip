import math

import torch

import e3nn.o3
import e3nn.nn

from nequip.nn import GraphModuleMixin
from nequip.utils import Config


# == Load old state ==
def initialize_from_state(config: Config, model: GraphModuleMixin, initialize: bool):
    """Initialize the model from the state dict file given by the config options `initial_model_state`.

    Only loads the state dict if `initialize` is `True`; this is meant for, say, starting a training from a previous state.

    If `initial_model_state_strict` controls
    > whether to strictly enforce that the keys in state_dict
    > match the keys returned by this module's state_dict() function

    See https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict.
    """
    if not initialize:
        return model  # do nothing
    return load_model_state(
        config=config, model=model, initialize=initialize, _prefix="initial_model_state"
    )


def load_model_state(
    config: Config,
    model: GraphModuleMixin,
    initialize: bool,
    _prefix: str = "load_model_state",
):
    """Load the model from the state dict file given by the config options `load_model_state`.

    Loads the state dict always; this is meant, for example, for building a new model to deploy with a given state dict.

    If `load_model_state_strict` controls
    > whether to strictly enforce that the keys in state_dict
    > match the keys returned by this module's state_dict() function

    See https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict.
    """
    if _prefix not in config:
        raise KeyError(
            f"initialize_from_state requires the `{_prefix}` option specifying the state to initialize from"
        )
    state = torch.load(config[_prefix])
    model.load_state_dict(state, strict=config.get(_prefix + "_strict", True))
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
