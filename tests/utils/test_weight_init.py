import torch

from nequip.utils.uniform_init import unit_uniform_init_


def test_unif_init():
    t = torch.empty(10_000)
    unit_uniform_init_(t)
    assert (t.square().mean() - 1.0).abs() <= 0.1
