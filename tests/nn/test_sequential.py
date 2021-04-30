import torch

from nequip.data import AtomicDataDict
from nequip.nn.embedding import OneHotAtomEncoding
from nequip.nn import SequentialGraphNetwork, AtomwiseLinear


def test_basic():
    sgn = SequentialGraphNetwork.from_parameters(
        shared_params={"num_species": 3},
        layers={"one_hot": OneHotAtomEncoding, "linear": AtomwiseLinear},
    )
    sgn(
        {
            AtomicDataDict.POSITIONS_KEY: torch.randn(5, 3),
            AtomicDataDict.EDGE_INDEX_KEY: torch.LongTensor([[0, 1], [1, 0]]),
            AtomicDataDict.SPECIES_INDEX_KEY: torch.LongTensor([0, 0, 1, 2, 0]),
        }
    )


def test_append():
    sgn = SequentialGraphNetwork.from_parameters(
        shared_params={"num_species": 3}, layers={"one_hot": OneHotAtomEncoding}
    )
    sgn.append_from_parameters(
        shared_params={"out_field": AtomicDataDict.NODE_FEATURES_KEY},
        name="linear",
        builder=AtomwiseLinear,
        params={"out_field": "thing"},
    )
    assert isinstance(sgn.linear, AtomwiseLinear)
    out = sgn(
        {
            AtomicDataDict.POSITIONS_KEY: torch.randn(5, 3),
            AtomicDataDict.EDGE_INDEX_KEY: torch.LongTensor([[0, 1], [1, 0]]),
            AtomicDataDict.SPECIES_INDEX_KEY: torch.LongTensor([0, 0, 1, 2, 0]),
        }
    )
    assert out["thing"].shape == out[AtomicDataDict.NODE_FEATURES_KEY].shape
