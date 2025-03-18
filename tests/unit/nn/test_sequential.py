import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.nn.embedding import NodeTypeEmbed
from nequip.nn import SequentialGraphNetwork, AtomwiseLinear


def test_basic():
    type_names = ["A", "B", "C"]
    node_type = NodeTypeEmbed(type_names=type_names, num_features=13)
    linear = AtomwiseLinear(irreps_in=node_type.irreps_out)

    sgn = SequentialGraphNetwork(
        {
            "one_hot": node_type,
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
    node_type = NodeTypeEmbed(type_names=["A", "B", "C"], num_features=13)
    sgn = SequentialGraphNetwork(
        modules={"node_type": node_type},
    )
    sgn.append(
        name="linear",
        module=AtomwiseLinear(out_field="thing", irreps_in=node_type.irreps_out),
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
    node_type = NodeTypeEmbed(type_names=["A", "B", "C"], num_features=13)
    linear = AtomwiseLinear(irreps_in=node_type.irreps_out)
    sgn = SequentialGraphNetwork({"node_type": node_type, "lin2": linear})
    keys = {"before": "lin2", "after": "node_type"}
    sgn.insert(
        name="lin1",
        module=AtomwiseLinear(
            out_field=AtomicDataDict.NODE_FEATURES_KEY, irreps_in=node_type.irreps_out
        ),
        **{mode: keys[mode]},
    )
    assert isinstance(sgn.lin1, AtomwiseLinear)
    assert len(sgn) == 3
    assert sgn[0] is sgn.node_type
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
