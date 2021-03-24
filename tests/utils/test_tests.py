import pytest

import torch

from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.utils.test import assert_AtomicData_equivariant


class BadModule(GraphModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._init_irreps(irreps_in={"x": o3.Irreps("4x1o")})

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        bad = torch.zeros(12)
        bad[2] = 7.898989
        data["x"] = data["x"] + bad
        return data


def test_equivar_test():
    badmod = BadModule()
    inp = {"x": badmod.irreps_in["x"].randn(2, -1)}
    with pytest.raises(AssertionError):
        assert_AtomicData_equivariant(badmod, data_in=inp)
