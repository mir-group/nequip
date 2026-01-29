# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
import packaging.version

from .version_utils import get_version_safe


_TORCH_VERSION = packaging.version.parse(get_version_safe(torch.__name__))
_TORCH_GE_2_4 = packaging.version.parse(
    _TORCH_VERSION.base_version
) >= packaging.version.parse("2.4")  # open equivariance's lowest version
_TORCH_GE_2_6 = packaging.version.parse(
    _TORCH_VERSION.base_version
) >= packaging.version.parse("2.6")
_TORCH_GE_2_7 = packaging.version.parse(
    _TORCH_VERSION.base_version
) >= packaging.version.parse("2.7")
_TORCH_GE_2_8 = packaging.version.parse(
    _TORCH_VERSION.base_version
) >= packaging.version.parse("2.8")
_TORCH_GE_2_9 = packaging.version.parse(
    _TORCH_VERSION.base_version
) >= packaging.version.parse("2.9")
_TORCH_GE_2_10 = packaging.version.parse(
    _TORCH_VERSION.base_version
) >= packaging.version.parse("2.10")
# used for skipping specific tests for PyTorch 2.10.0 due to known bugs
# (specifically CPU+aotinductor compilation issues)
# this is not GE in case future PyTorch versions resolve the issue
_TORCH_IS_2_10_0 = packaging.version.parse(
    _TORCH_VERSION.base_version
) == packaging.version.parse("2.10.0")


def check_pt2_compile_compatibility():
    assert _TORCH_GE_2_6, (
        f"PyTorch >= 2.6 required for PT2 compilation functionality, "
        f"but {_TORCH_VERSION} found."
    )
