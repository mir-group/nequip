# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Union
import contextlib
import contextvars

import torch

from e3nn import set_optimization_defaults, get_optimization_defaults

_CONDITIONAL_TORCHSCRIPT_MODE = contextvars.ContextVar(
    "_CONDITIONAL_TORCHSCRIPT_MODE", default=True
)


@contextlib.contextmanager
def conditional_torchscript_mode(enabled: bool):
    global _CONDITIONAL_TORCHSCRIPT_MODE
    # save previous state
    init_val_e3nn = get_optimization_defaults()["jit_script_fx"]
    init_val_here = _CONDITIONAL_TORCHSCRIPT_MODE.get()
    # set mode variables
    set_optimization_defaults(jit_script_fx=enabled)
    _CONDITIONAL_TORCHSCRIPT_MODE.set(enabled)
    yield
    # restore state
    set_optimization_defaults(jit_script_fx=init_val_e3nn)
    _CONDITIONAL_TORCHSCRIPT_MODE.set(init_val_here)


def conditional_torchscript_jit(
    module: torch.nn.Module,
) -> Union[torch.jit.ScriptModule, torch.nn.Module]:
    """Compile a module with TorchScript, conditional on whether it is enabled by ``conditional_torchscript_mode``"""
    global _CONDITIONAL_TORCHSCRIPT_MODE
    if _CONDITIONAL_TORCHSCRIPT_MODE.get():
        return torch.jit.script(module)
    else:
        return module


def prepare_model_for_compile(model, device):
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    model.to(device)
    return model
