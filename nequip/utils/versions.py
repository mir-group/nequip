# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
import e3nn
import nequip

from ..__init__ import _DISCOVERED_NEQUIP_EXTENSION
from .logger import RankedLogger
from typing import Tuple
import importlib
import packaging.version

logger = RankedLogger(__name__, rank_zero_only=True)

_TORCH_VERSION = packaging.version.parse(torch.__version__)
_TORCH_GE_2_6 = packaging.version.parse(
    _TORCH_VERSION.base_version
) >= packaging.version.parse("2.6")


def check_pt2_compile_compatibility():
    assert (
        _TORCH_GE_2_6
    ), f"PyTorch >= 2.6 required for PT2 compilation functionality, but {_TORCH_VERSION} found."


# get versions of torch, e3nn, nequip and all extension packages
_DEFAULT_VERSION_CODES = [torch, e3nn, nequip]
for ep in _DISCOVERED_NEQUIP_EXTENSION:
    _DEFAULT_VERSION_CODES.append(importlib.import_module(ep.value))


def get_current_code_versions(verbose=True) -> Tuple[dict, dict]:
    code_versions = {}
    for code in _DEFAULT_VERSION_CODES:
        code_versions[code.__name__] = str(code.__version__)

    if verbose:
        logger.info("Version Information:")
        for k, v in code_versions.items():
            logger.info(f"{k} {v}")

    return code_versions
