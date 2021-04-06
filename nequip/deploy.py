from typing import Union, Final

import torch

from e3nn.util.jit import script

import nequip
from nequip.nn import GraphModuleMixin


R_MAX_KEY: Final[str] = "r_max"
ORIG_CONFIG_KEY: Final[str] = "orig_config"
NEQUIP_VERSION_KEY: Final[str] = "nequip_version"


def make_model_deployable(
    model: Union[GraphModuleMixin, torch.jit.ScriptModule],
) -> torch.jit.ScriptModule:
    if getattr(model, NEQUIP_VERSION_KEY, None) is not None:
        raise ValueError("This model has already been prepared for deployment.")

    if isinstance(model, GraphModuleMixin):
        # need to compile
        model = script(model)
    elif isinstance(model, torch.jit.ScriptModule):
        pass
    else:
        raise TypeError(f"Model `{model}` is not a (compiled) GraphModuleMixin.")

    # store version of nequip
    setattr(model, NEQUIP_VERSION_KEY, nequip.__version__)
    # eval mode
    model = model.eval()

    return model
