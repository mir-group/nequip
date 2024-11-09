import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.nn.embedding import OneHotAtomEncoding
from nequip.nn import SequentialGraphNetwork, AtomwiseLinear


def test_basic():
    type_names = ["A", "B", "C"]
    one_hot = OneHotAtomEncoding(type_names=type_names)
    linear = AtomwiseLinear(irreps_in=one_hot.irreps_out)

    sgn = SequentialGraphNetwork(
        {
            "one_hot": one_hot,
            "linear": linear,
        }
    )
    sgn(
        {
            AtomicDataDict.POSITIONS_KEY: torch.randn(5, 3),
            AtomicDataDict.EDGE_INDEX_KEY: torch.LongTensor([[0, 1], [1, 0]]),
            AtomicDataDict.ATOM_TYPE_KEY: torch.LongTensor([0, 0, 1, 2, 0]),
        }
    )


def test_append():
    one_hot = OneHotAtomEncoding(type_names=["A", "B", "C"])
    sgn = SequentialGraphNetwork(
        modules={"one_hot": one_hot},
    )
    sgn.append(
        name="linear",
        module=AtomwiseLinear(out_field="thing", irreps_in=one_hot.irreps_out),
    )
    assert isinstance(sgn.linear, AtomwiseLinear)
    out = sgn(
        {
            AtomicDataDict.POSITIONS_KEY: torch.randn(5, 3),
            AtomicDataDict.EDGE_INDEX_KEY: torch.LongTensor([[0, 1], [1, 0]]),
            AtomicDataDict.ATOM_TYPE_KEY: torch.LongTensor([0, 0, 1, 2, 0]),
        }
    )
    assert out["thing"].shape == out[AtomicDataDict.NODE_FEATURES_KEY].shape


@pytest.mark.parametrize("mode", {"before", "after"})
def test_insert(mode):
    one_hot = OneHotAtomEncoding(type_names=["A", "B", "C"])
    linear = AtomwiseLinear(irreps_in=one_hot.irreps_out)
    sgn = SequentialGraphNetwork({"one_hot": one_hot, "lin2": linear})
    keys = {"before": "lin2", "after": "one_hot"}
    sgn.insert(
        name="lin1",
        module=AtomwiseLinear(
            out_field=AtomicDataDict.NODE_FEATURES_KEY, irreps_in=one_hot.irreps_out
        ),
        **{mode: keys[mode]},
    )
    assert isinstance(sgn.lin1, AtomwiseLinear)
    assert len(sgn) == 3
    assert sgn[0] is sgn.one_hot
    assert sgn[1] is sgn.lin1
    assert sgn[2] is sgn.lin2
    out = sgn(
        {
            AtomicDataDict.POSITIONS_KEY: torch.randn(5, 3),
            AtomicDataDict.EDGE_INDEX_KEY: torch.LongTensor([[0, 1], [1, 0]]),
            AtomicDataDict.ATOM_TYPE_KEY: torch.LongTensor([0, 0, 1, 2, 0]),
        }
    )
    assert AtomicDataDict.NODE_FEATURES_KEY in out
