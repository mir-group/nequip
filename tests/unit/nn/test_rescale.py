import pytest

import sys

if sys.version_info[1] >= 7:
    import contextlib
else:
    # has backport of nullcontext
    import contextlib2 as contextlib

import torch

from e3nn.util.test import assert_auto_jitable

from nequip.nn import RescaleOutput, GraphModel
from nequip.nn.embedding import OneHotAtomEncoding
from nequip.data import AtomicDataDict, AtomicData
from nequip.utils import dtype_from_name, torch_default_dtype
from nequip.utils.test import assert_AtomicData_equivariant


@pytest.mark.parametrize("scale_by", [0.77, 1.0, None])
@pytest.mark.parametrize("shift_by", [0.0, 0.4443, None])
@pytest.mark.parametrize("scale_trainable", [True, False])
@pytest.mark.parametrize("shift_trainable", [True, False])
def test_rescale(
    CH3CHO,
    scale_by,
    shift_by,
    scale_trainable,
    shift_trainable,
    model_dtype,
):
    _, data = CH3CHO
    with torch_default_dtype(dtype_from_name(model_dtype)):
        oh = GraphModel(
            OneHotAtomEncoding(
                num_types=3,
                irreps_in=data.irreps,
            ),
            model_dtype=dtype_from_name(model_dtype),
        )

    # some combinations are illegal and should raise
    build_with = contextlib.nullcontext()
    if scale_by is None and scale_trainable:
        build_with = pytest.raises(ValueError)
    elif shift_by is None and shift_trainable:
        build_with = pytest.raises(ValueError)

    rescale = None
    with build_with:
        rescale = RescaleOutput(
            model=oh,
            scale_keys=AtomicDataDict.NODE_ATTRS_KEY,
            shift_keys=AtomicDataDict.NODE_ATTRS_KEY,
            scale_by=scale_by,
            shift_by=shift_by,
            scale_trainable=scale_trainable,
            shift_trainable=shift_trainable,
        )

    if rescale is None:
        return

    # == Check basics ==
    assert_auto_jitable(rescale)
    for training_mode in [True, False]:
        rescale.train(training_mode)
        rescale(AtomicData.to_AtomicDataDict(data))
        assert_AtomicData_equivariant(rescale, data)

    # == Check scale/shift ==
    for training_mode in [True, False]:
        rescale.train(training_mode)

        # oh_out is in model_dtype
        oh_out = oh(AtomicData.to_AtomicDataDict(data))[AtomicDataDict.NODE_ATTRS_KEY]
        # rescale_out is in default_dtype
        rescale_out = rescale(AtomicData.to_AtomicDataDict(data))[
            AtomicDataDict.NODE_ATTRS_KEY
        ]

        # only default_dtype=float64 will be tested (see above pytest.skip check)
        # model_dtype can be float32 or float64
        # so we cast oh_out (model_dtype) to rescale_out (model_dtype) for testing
        oh_out = oh_out.to(dtype=rescale_out.dtype)

        if training_mode:
            assert torch.all(oh_out == rescale_out)
            continue  # don't test anything else
        # we are now in eval mode if here, test rescaling
        # node attrs are a one hot, so we know orig then are zeros and ones
        if scale_by is None and shift_by is None:
            assert torch.all(oh_out == rescale_out)
        if shift_by is None:
            # no shift preserves zeros
            assert torch.all((rescale_out == 0.0) == (oh_out == 0.0))
        if scale_by is None and shift_by is not None:
            # check that difference is right
            assert torch.allclose(rescale_out - oh_out, torch.as_tensor(shift_by))
        if scale_by is not None and shift_by is None:
            # check that ratio is right
            ratio = torch.nan_to_num(rescale_out / oh_out)
            assert torch.allclose(ratio[oh_out != 0.0], torch.as_tensor(scale_by))
        if scale_by is not None and shift_by is not None:
            assert torch.allclose(rescale_out, scale_by * oh_out + shift_by)
