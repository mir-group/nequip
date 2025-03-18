import pytest

import torch

from e3nn.util.test import assert_auto_jitable

from nequip.nn import RescaleOutput, GraphModel
from nequip.nn.embedding import NodeTypeEmbed
from nequip.data import AtomicDataDict
from nequip.utils import dtype_from_name, torch_default_dtype
from nequip.utils.test import assert_AtomicData_equivariant


@pytest.mark.parametrize("scale_by", [0.77, 1.0, 2.6])
def test_rescale(
    CH3CHO,
    scale_by,
    model_dtype,
):
    _, data = CH3CHO
    with torch_default_dtype(dtype_from_name(model_dtype)):
        oh = GraphModel(
            NodeTypeEmbed(
                type_names=["A", "B", "C"],
                num_features=13,
            ),
        )

    rescale = RescaleOutput(
        model=oh,
        scale_keys=AtomicDataDict.NODE_ATTRS_KEY,
        scale_by=scale_by,
    )

    # == Check basics ==
    assert_auto_jitable(rescale)
    for training_mode in [True, False]:
        rescale.train(training_mode)
        rescale(data)
        assert_AtomicData_equivariant(rescale, data)

    # == Check scale/shift ==
    # oh_out is in model_dtype
    oh_out = oh(data)[AtomicDataDict.NODE_ATTRS_KEY]
    # rescale_out is in default_dtype
    rescale_out = rescale(data)[AtomicDataDict.NODE_ATTRS_KEY]

    # model_dtype can be float32 or float64
    # so we cast oh_out (model_dtype) to rescale_out (model_dtype) for testing
    oh_out = oh_out.to(dtype=rescale_out.dtype)

    # node attrs are a one hot, so we know orig then are zeros and ones
    ratio = torch.nan_to_num(rescale_out / oh_out)
    assert torch.allclose(ratio[oh_out != 0.0], torch.as_tensor(scale_by))
