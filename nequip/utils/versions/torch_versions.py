# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
import packaging.version

from .version_utils import get_version_safe


_TORCH_VERSION = packaging.version.parse(get_version_safe(torch.__name__))
_TORCH_GE_2_4 = packaging.version.parse(
    _TORCH_VERSION.base_version
) >= packaging.version.parse(
    "2.4"
)  # open equivariance's lowest version
_TORCH_GE_2_6 = packaging.version.parse(
    _TORCH_VERSION.base_version
) >= packaging.version.parse("2.6")
_TORCH_GE_2_8 = packaging.version.parse(
    _TORCH_VERSION.base_version
) >= packaging.version.parse("2.8")


def check_pt2_compile_compatibility():
    assert _TORCH_GE_2_6, (
        f"PyTorch >= 2.6 required for PT2 compilation functionality, "
        f"but {_TORCH_VERSION} found."
    )
