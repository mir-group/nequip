import math

import torch

import e3nn.o3
import e3nn.nn


def unit_uniform_init_(t: torch.Tensor):
    t.uniform_(-math.sqrt(3), math.sqrt(3))


def uniform_initialize(
    mod: torch.nn.Module, do_fc: bool = True, do_linear: bool = True, do_tp: bool = True
) -> None:
    if do_fc and isinstance(mod, e3nn.nn.FullyConnectedNet):
        for w in mod.weights:
            unit_uniform_init_(w)
    elif do_linear and isinstance(mod, e3nn.o3.Linear) and mod.internal_weights:
        unit_uniform_init_(mod.weight)
    elif do_tp and isinstance(mod, e3nn.o3.TensorProduct) and mod.internal_weights:
        unit_uniform_init_(mod.weight)
    return
