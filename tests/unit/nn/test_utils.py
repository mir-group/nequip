import torch

from nequip.data import AtomicDataDict
from nequip.nn.embedding import OneHotAtomEncoding
from nequip.nn import SequentialGraphNetwork, SaveForOutput, AtomwiseLinear, GraphModel
from nequip.utils import dtype_from_name, torch_default_dtype


def test_basic(model_dtype):
    type_names = ["A", "B", "C", "D"]
    with torch_default_dtype(dtype_from_name(model_dtype)):

        one_hot = OneHotAtomEncoding(type_names=type_names)
        save = SaveForOutput(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field="saved",
            irreps_in=one_hot.irreps_out,
        )
        linear = AtomwiseLinear(irreps_in=save.irreps_out)
        model = GraphModel(
            SequentialGraphNetwork(
                {
                    "one_hot": one_hot,
                    "save": save,
                    "linear": linear,
                }
            ),
            model_dtype=dtype_from_name(model_dtype),
            type_names=["A", "B", "C", "D"],
        )
    out = model(
        {
            AtomicDataDict.POSITIONS_KEY: torch.randn(5, 3),
            AtomicDataDict.EDGE_INDEX_KEY: torch.LongTensor([[0, 1], [1, 0]]),
            AtomicDataDict.ATOM_TYPE_KEY: torch.LongTensor([0, 0, 1, 2, 0]),
        }
    )
    saved = out["saved"]
    assert saved.shape == (5, 4)
    assert torch.all(saved[0] == torch.as_tensor([1.0, 0.0, 0.0, 0.0]))
