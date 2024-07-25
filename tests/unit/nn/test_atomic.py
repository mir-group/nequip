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


@pytest.fixture(scope="class", params=[0, 1, 2])
def model(model_dtype, request):
    zero_species = request.param
    shifts = [3, 5, 7]
    shifts[zero_species] = 0
    params = dict(
        type_names=["A", "B", "C"],
        total_shift=1.0,
        shifts=shifts,
    )

    with torch_default_dtype(dtype_from_name(model_dtype)):
        model = GraphModel(
            SequentialGraphNetwork.from_parameters(
                shared_params=params,
                layers={
                    "one_hot": OneHotAtomEncoding,
                    "linear": (
                        AtomwiseLinear,
                        dict(irreps_out="1x0e", out_field="e"),
                    ),
                    "shift": (
                        PerTypeScaleShift,
                        dict(
                            field="e",
                            out_field="shifted",
                            scales=1.0,
                            shifts=0.0,
                            arguments_in_dataset_units=False,
                            default_dtype=torch.get_default_dtype(),
                        ),
                    ),
                    "sum": (
                        AtomwiseReduce,
                        dict(reduce="sum", field="shifted", out_field="sum"),
                    ),
                },
            ),
            model_dtype=dtype_from_name(model_dtype),
            type_names=["A", "B", "C"],
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
    print(result1["shifted"], result2["shifted"], result12["shifted"])
