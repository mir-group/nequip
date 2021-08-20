import torch

from nequip.data import AtomicDataDict
from nequip.nn.embedding import OneHotAtomEncoding
from nequip.nn import SequentialGraphNetwork, SaveForOutput, AtomwiseLinear


def test_basic():
    sgn = SequentialGraphNetwork.from_parameters(
        shared_params={"num_species": 4},
        layers={
            "one_hot": OneHotAtomEncoding,
            "save": (
                SaveForOutput,
                dict(field=AtomicDataDict.NODE_FEATURES_KEY, out_field="saved"),
            ),
            "linear": AtomwiseLinear,
        },
    )
    out = sgn(
        {
            AtomicDataDict.POSITIONS_KEY: torch.randn(5, 3),
            AtomicDataDict.EDGE_INDEX_KEY: torch.LongTensor([[0, 1], [1, 0]]),
            AtomicDataDict.ATOM_TYPE_KEY: torch.LongTensor([0, 0, 1, 2, 0]),
        }
    )
    saved = out["saved"]
    assert saved.shape == (5, 4)
    assert torch.all(saved[0] == torch.as_tensor([1.0, 0.0, 0.0, 0.0]))
