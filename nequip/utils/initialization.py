import math

import torch

import e3nn.o3
import e3nn.nn


# == Uniform init ==
def unit_uniform_init_(t: torch.Tensor):
    t.uniform_(-math.sqrt(3), math.sqrt(3))


def uniform_initialize_fcs(mod: torch.nn.Module):
    if isinstance(mod, e3nn.nn.FullyConnectedNet):
        for w in mod.weights:
            unit_uniform_init_(w)
    # no need to do torch.nn.Linear, which is uniform by default


def uniform_initialize_linears(mod: torch.nn.Module):
    if isinstance(mod, e3nn.o3.Linear) and mod.internal_weights:
        unit_uniform_init_(mod.weight)


def uniform_initialize_tps(mod: torch.nn.Module):
    if isinstance(mod, e3nn.o3.TensorProduct) and mod.internal_weights:
        unit_uniform_init_(mod.weight)


# == Xavier ==
def xavier_initialize_fcs(mod: torch.nn.Module):
    if isinstance(mod, e3nn.nn.FullyConnectedNet):
        for w in mod.weights:
            # in FC:
            # h_in, _h_out = W.shape
            # W = W / h_in**0.5
            torch.nn.init.xavier_uniform_(w, gain=w.shape[0] ** 0.5)
    elif isinstance(mod, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(mod.weight)


# == Orthogonal ==
def unit_orthogonal_init_(t: torch.Tensor):
    """Orthogonal init with <x_i^2> = N"""
    assert t.ndim == 2
    torch.nn.init.orthogonal_(t, gain=math.sqrt(max(t.shape)))


def orthogonal_initialize_linears(mod: torch.nn.Module):
    if isinstance(mod, e3nn.o3.Linear) and mod.internal_weights:
        for w in mod.weight_views():
            unit_uniform_init_(w)
