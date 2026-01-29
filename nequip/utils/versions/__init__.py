# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

from .torch_versions import (
    _TORCH_GE_2_4,
    _TORCH_GE_2_6,
    _TORCH_GE_2_7,
    _TORCH_GE_2_8,
    _TORCH_GE_2_9,
    _TORCH_GE_2_10,
    _TORCH_IS_2_10_0,
    check_pt2_compile_compatibility,
)

from .package_versions import get_current_code_versions

__all__ = [
    "_TORCH_GE_2_4",
    "_TORCH_GE_2_6",
    "_TORCH_GE_2_7",
    "_TORCH_GE_2_8",
    "_TORCH_GE_2_9",
    "_TORCH_GE_2_10",
    "_TORCH_IS_2_10_0",
    "check_pt2_compile_compatibility",
    "get_current_code_versions",
]
