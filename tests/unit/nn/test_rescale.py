import pytest

import torch

from e3nn.util.test import assert_auto_jitable

from nequip.nn import RescaleOutput, GraphModel
from nequip.nn.embedding import OneHotAtomEncoding
from nequip.data import AtomicDataDict
from nequip.utils import dtype_from_name, torch_default_dtype
from nequip.utils.test import assert_AtomicData_equivariant


@pytest.mark.parametrize("scale_by", [0.77, 1.0, None])
@pytest.mark.parametrize("shift_trainable", [True, False])
def test_rescale(
    CH3CHO,
    scale_by,
    shift_trainable,
    model_dtype,
):
    _, data = CH3CHO
    with torch_default_dtype(dtype_from_name(model_dtype)):
        oh = GraphModel(
            OneHotAtomEncoding(
                type_names=["A", "B", "C"],
            ),
            model_dtype=dtype_from_name(model_dtype),
            type_names=["A", "B", "C"],
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

    # only default_dtype=float64 will be tested (see above pytest.skip check)
    # model_dtype can be float32 or float64
    # so we cast oh_out (model_dtype) to rescale_out (model_dtype) for testing
    oh_out = oh_out.to(dtype=rescale_out.dtype)

    # node attrs are a one hot, so we know orig then are zeros and ones
    if scale_by is None:
        assert torch.all(oh_out == rescale_out)
    else:
        ratio = torch.nan_to_num(rescale_out / oh_out)
        assert torch.allclose(ratio[oh_out != 0.0], torch.as_tensor(scale_by))
