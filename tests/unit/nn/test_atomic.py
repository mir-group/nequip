import pytest
import torch

from nequip.data import AtomicData
from nequip.nn import (
    AtomwiseLinear,
    AtomwiseReduce,
    PerSpeciesScaleShift,
    SequentialGraphNetwork,
)
from nequip.nn.embedding import (
    OneHotAtomEncoding,
)
from nequip.utils.torch_geometric import Batch


@pytest.fixture(scope="class", params=[0, 1, 2])
def model(float_tolerance, request):
    zero_species = request.param
    shifts = [3, 5, 7]
    shifts[zero_species] = 0
    params = dict(
        num_types=3,
        total_shift=1.0,
        shifts=shifts,
    )
    return SequentialGraphNetwork.from_parameters(
        shared_params=params,
        layers={
            "one_hot": OneHotAtomEncoding,
            "linear": (
                AtomwiseLinear,
                dict(irreps_out="1x0e", out_field="e"),
            ),
            "shift": (
                PerSpeciesScaleShift,
                dict(
                    field="e",
                    out_field="shifted",
                    scales=1.0,
                    shifts=0.0,
                    arguments_in_dataset_units=False,
                ),
            ),
            "sum": (
                AtomwiseReduce,
                dict(field="shifted", out_field="sum"),
            ),
        },
    )


@pytest.fixture(scope="class")
def batches(float_tolerance, nequip_dataset):
    b = []
    for idx in [[0], [1], [0, 1]]:
        b += [AtomicData.to_AtomicDataDict(Batch.from_data_list(nequip_dataset[idx]))]
    return b


def test_per_species_shift(nequip_dataset, batches, model):
    batch1, batch2, batch12 = batches
    result1 = model(batch1)
    result2 = model(batch2)
    result12 = model(batch12)

    assert torch.isclose(result1["sum"], result12["sum"][0])
    assert torch.isclose(result2["sum"], result12["sum"][1])
    print(result1["shifted"], result2["shifted"], result12["shifted"])
