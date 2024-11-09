import pytest
import torch

from nequip.data import AtomicDataDict
from nequip.nn import (
    AtomwiseLinear,
    AtomwiseReduce,
    PerTypeScaleShift,
    SequentialGraphNetwork,
    GraphModel,
)
from nequip.nn.embedding import (
    OneHotAtomEncoding,
)
from nequip.utils import dtype_from_name, torch_default_dtype


@pytest.fixture(
    scope="class",
    params=[
        (1.3, 5.7),
        (1.3, [5.7, 77.7, 34.1]),
        ([1.3, 9.3, 4.1], 5.7),
        ([1.3, 9.3, 4.1], [5.7, 77.7, 34.1]),
    ],
)
def model(model_dtype, request):
    scales, shifts = request.param
    type_names = ["A", "B", "C"]
    with torch_default_dtype(dtype_from_name(model_dtype)):
        oh = OneHotAtomEncoding(type_names=type_names)
        linear = AtomwiseLinear(
            irreps_out="1x0e",
            out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            irreps_in=oh.irreps_out,
        )
        shift = PerTypeScaleShift(
            type_names=type_names,
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            scales=scales,
            shifts=shifts,
            irreps_in=linear.irreps_out,
        )
        sum_reduce = AtomwiseReduce(
            reduce="sum",
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field="sum",
            irreps_in=shift.irreps_out,
        )

        model = GraphModel(
            SequentialGraphNetwork(
                {
                    "one_hot": oh,
                    "linear": linear,
                    "shift": shift,
                    "sum": sum_reduce,
                },
            ),
            model_dtype=dtype_from_name(model_dtype),
            type_names=type_names,
        )
    return model


@pytest.fixture(scope="class")
def batches(nequip_dataset):
    print(nequip_dataset[0])
    print(nequip_dataset[1])
    b = []
    for idx in [[0], [1], [0, 1]]:
        b += [AtomicDataDict.batched_from_list(nequip_dataset[idx])]
    return b


def test_per_species_shift(nequip_dataset, batches, model):
    batch1, batch2, batch12 = batches
    result1 = model(batch1)
    result2 = model(batch2)
    result12 = model(batch12)

    assert torch.isclose(result1["sum"], result12["sum"][0])
    assert torch.isclose(result2["sum"], result12["sum"][1])
