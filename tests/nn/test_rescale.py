import pytest

import contextlib

import torch

from e3nn.util.test import assert_auto_jitable

from nequip.nn import RescaleOutput
from nequip.data import AtomicDataDict, AtomicData
from nequip.utils.test import assert_AtomicData_equivariant
from nequip.nn.embedding import OneHotAtomEncoding


@pytest.mark.parametrize("scale_by", [0.77, 1.0, None])
@pytest.mark.parametrize("shift_by", [0.0, 0.4443, None])
@pytest.mark.parametrize("trainable_global_rescale_scale", [True, False])
@pytest.mark.parametrize("trainable_global_rescale_shift", [True, False])
def test_rescale(
    CH3CHO,
    scale_by,
    shift_by,
    trainable_global_rescale_scale,
    trainable_global_rescale_shift,
):
    _, data = CH3CHO
    oh = OneHotAtomEncoding(
        allowed_species=torch.unique(data[AtomicDataDict.ATOMIC_NUMBERS_KEY]),
        irreps_in=data.irreps,
    )

    # some combinations are illegal and should raise
    build_with = contextlib.nullcontext()
    if scale_by is None and trainable_global_rescale_scale:
        build_with = pytest.raises(ValueError)
    elif shift_by is None and trainable_global_rescale_shift:
        build_with = pytest.raises(ValueError)

    rescale = None
    with build_with:
        rescale = RescaleOutput(
            model=oh,
            scale_keys=AtomicDataDict.NODE_ATTRS_KEY,
            shift_keys=AtomicDataDict.NODE_ATTRS_KEY,
            scale_by=scale_by,
            shift_by=shift_by,
            trainable_global_rescale_scale=trainable_global_rescale_scale,
            trainable_global_rescale_shift=trainable_global_rescale_shift,
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
        oh_out = oh(AtomicData.to_AtomicDataDict(data))[AtomicDataDict.NODE_ATTRS_KEY]
        rescale_out = rescale(AtomicData.to_AtomicDataDict(data))[
            AtomicDataDict.NODE_ATTRS_KEY
        ]
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
