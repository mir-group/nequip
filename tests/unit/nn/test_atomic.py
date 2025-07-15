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
from nequip.nn.embedding import NodeTypeEmbed
from nequip.utils import dtype_from_name, torch_default_dtype
from nequip.model.modify_utils import modify


@pytest.fixture(
    scope="class",
    params=[
        (1.3, 5.7),
        (1.3, {"A": 5.7, "B": 77.7, "C": 34.1}),
        ({"A": 1.3, "B": 9.3, "C": 4.1}, 5.7),
        ({"A": 1.3, "B": 9.3, "C": 4.1}, {"A": 5.7, "B": 77.7, "C": 34.1}),
    ],
)
def model(model_dtype, request):
    scales, shifts = request.param
    type_names = ["A", "B", "C"]
    with torch_default_dtype(dtype_from_name(model_dtype)):
        nt = NodeTypeEmbed(type_names=type_names, num_features=13)
        linear = AtomwiseLinear(
            irreps_out="1x0e",
            out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            irreps_in=nt.irreps_out,
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
                    "node_type": nt,
                    "linear": linear,
                    "shift": shift,
                    "sum": sum_reduce,
                },
            ),
        )
    return model


@pytest.fixture(scope="class")
def batches(nequip_dataset):
    """Create test batches for single frames and combined batches."""
    b = []
    for idx in [[0], [1], [0, 1]]:
        b += [AtomicDataDict.batched_from_list(nequip_dataset[idx])]
    return b


def test_per_species_shift(nequip_dataset, batches, model):
    batch1, batch2, batch12 = batches
    result1 = model(batch1)
    result2 = model(batch2)
    result12 = model(batch12)

    tol = {torch.float32: 1e-6, torch.float64: 1e-12}[model.model_dtype]
    torch.testing.assert_close(
        result1["sum"].view(-1), result12["sum"][0].view(-1), atol=tol, rtol=tol
    )
    torch.testing.assert_close(
        result2["sum"].view(-1), result12["sum"][1].view(-1), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("scales", [None, {"A": 1.5}, {"A": 1.5, "B": 2.5, "C": 0.8}])
@pytest.mark.parametrize("shifts", [None, {"A": 0.2}, {"A": 0.2, "B": -0.3, "C": 0.1}])
@pytest.mark.parametrize("scales_trainable", [True, False])
@pytest.mark.parametrize("shifts_trainable", [True, False])
def test_modify_per_type_scale_shift(
    batches, model, scales, shifts, scales_trainable, shifts_trainable
):
    """Test modifying PerTypeScaleShift parameters and trainable flags."""
    batch1, _, _ = batches

    # get original result
    original_result = model(batch1)

    # build modification arguments
    modifier_args = {
        "modifier": "modify_PerTypeScaleShift",
        "scales": scales,
        "shifts": shifts,
        "scales_trainable": scales_trainable,
        "shifts_trainable": shifts_trainable,
    }
    modified_model = modify(model, [modifier_args])

    # check that modifications were applied
    shift_module = None
    for module in modified_model.modules():
        if isinstance(module, PerTypeScaleShift):
            shift_module = module
            break

    assert shift_module is not None

    # check that parameters are trainable as expected
    trainable_params = [p for p in shift_module.parameters() if p.requires_grad]
    if scales_trainable or shifts_trainable:
        assert len(trainable_params) > 0
    else:
        assert len(trainable_params) == 0

    # test that modified model produces different results when parameters are modified
    modified_result = modified_model(batch1)
    tol = {torch.float32: 1e-6, torch.float64: 1e-12}[model.model_dtype]
    if scales is not None or shifts is not None:
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                original_result["sum"], modified_result["sum"], atol=tol, rtol=tol
            )
    else:
        # when only trainable flags change, output should be the same
        torch.testing.assert_close(
            original_result["sum"], modified_result["sum"], atol=tol, rtol=tol
        )
