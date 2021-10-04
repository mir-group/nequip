import pytest

import torch

from e3nn import o3

from nequip.data import AtomicDataDict, register_fields, deregister_fields
from nequip.nn import GraphModuleMixin
from nequip.utils.test import (
    assert_AtomicData_equivariant,
    assert_permutation_equivariant,
)


class BadModule(GraphModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._init_irreps(irreps_in={AtomicDataDict.POSITIONS_KEY: "1x1o"})

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        bad = torch.zeros_like(data[AtomicDataDict.POSITIONS_KEY])
        bad[:, 2] = 7.898989
        data[AtomicDataDict.POSITIONS_KEY] = data[AtomicDataDict.POSITIONS_KEY] + bad
        return data


class BadPermuteModule(GraphModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._init_irreps(
            irreps_in={AtomicDataDict.POSITIONS_KEY: "1x1o"},
            irreps_out={AtomicDataDict.TOTAL_ENERGY_KEY: "0e"},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        factors = torch.ones(len(data[AtomicDataDict.POSITIONS_KEY]))
        factors[-1] = 7.8
        factors[0] = -4.5
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = (
            (data[AtomicDataDict.POSITIONS_KEY].norm(dim=-1).abs() * factors)
            .sum()
            .reshape((1, 1))
        )
        return data


class GoodModule(GraphModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._init_irreps(
            irreps_in={
                AtomicDataDict.POSITIONS_KEY: "1x1o",
                AtomicDataDict.NODE_FEATURES_KEY: "4x0e",
            },
            irreps_out={AtomicDataDict.NODE_FEATURES_KEY: "0e"},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[AtomicDataDict.NODE_FEATURES_KEY] = data[
            AtomicDataDict.NODE_FEATURES_KEY
        ].sum(dim=-1, keepdim=True)
        return data


class BadIrrepsModule(GraphModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._init_irreps(irreps_out={"x": o3.Irreps("4x2e")})

    def forward(self):
        return {"x": torch.randn(3, 5, 2)}  # wrong dims!


def test_equivar_fail():
    badmod = BadModule()
    inp = {
        AtomicDataDict.POSITIONS_KEY: badmod.irreps_in[
            AtomicDataDict.POSITIONS_KEY
        ].randn(2, -1),
        AtomicDataDict.EDGE_INDEX_KEY: torch.randint(0, 2, (2, 3)),
    }
    with pytest.raises(AssertionError):
        assert_AtomicData_equivariant(badmod, data_in=inp)


def test_equivar_test():
    mod = GoodModule()
    inp = {
        AtomicDataDict.POSITIONS_KEY: mod.irreps_in[AtomicDataDict.POSITIONS_KEY].randn(
            2, -1
        ),
        AtomicDataDict.EDGE_INDEX_KEY: torch.randint(0, 2, (2, 3)),
        AtomicDataDict.NODE_FEATURES_KEY: mod.irreps_in[
            AtomicDataDict.NODE_FEATURES_KEY
        ].randn(2, -1),
    }
    assert_AtomicData_equivariant(mod, data_in=inp)


def test_permute_fail():
    mod = BadPermuteModule()
    natom = 10
    inp = {
        AtomicDataDict.POSITIONS_KEY: mod.irreps_in[AtomicDataDict.POSITIONS_KEY].randn(
            natom, -1
        ),
        AtomicDataDict.EDGE_INDEX_KEY: torch.randint(0, natom, (2, 13)),
    }
    with pytest.raises(AssertionError):
        assert_permutation_equivariant(mod, inp)


def test_debug_mode():
    # Note that debug mode is enabled by default in the tests,
    # so there's nothing to enable
    badmod = BadIrrepsModule()
    with pytest.raises(ValueError):
        badmod()


class EdgePermuteModule(GraphModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._init_irreps(
            irreps_out={
                "my_edge": "1o",
            }
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data["my_edge"] = data[AtomicDataDict.POSITIONS_KEY][
            data[AtomicDataDict.EDGE_INDEX_KEY][0]
        ]
        return data


def test_permute_register():
    # Get a clean slate
    deregister_fields("my_edge")

    mod = EdgePermuteModule()
    n_pos = 20
    n_edge = 20
    inp = {
        AtomicDataDict.POSITIONS_KEY: mod.irreps_in[AtomicDataDict.POSITIONS_KEY].randn(
            n_pos, -1
        ),
        AtomicDataDict.EDGE_INDEX_KEY: torch.randint(0, n_pos, (2, n_edge)),
    }

    with pytest.raises(AssertionError):
        # Fails because thinks "my_edge" is invariant
        assert_permutation_equivariant(mod, data_in=dict(inp))
    assert_permutation_equivariant(
        mod, data_in=dict(inp), extra_edge_permute_fields=["my_edge"]
    )
    register_fields(edge_permute_fields=["my_edge"])
    assert_permutation_equivariant(mod, data_in=dict(inp))
