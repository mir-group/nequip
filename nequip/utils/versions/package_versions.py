# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
import e3nn
import nequip

from ...__init__ import _DISCOVERED_NEQUIP_EXTENSION
from ...utils.logger import RankedLogger
import importlib

from .version_utils import get_version_safe

logger = RankedLogger(__name__, rank_zero_only=True)


# get versions of torch, e3nn, nequip and all extension packages
_DEFAULT_VERSION_CODES = [torch, e3nn, nequip]

for ep in _DISCOVERED_NEQUIP_EXTENSION:
    _DEFAULT_VERSION_CODES.append(importlib.import_module(ep.value))


def get_current_code_versions(verbose: bool = True) -> dict:
    """Get versions of all relevant packages safely.

    Args:
        verbose (bool): whether to log version information

    Returns:
        Dictionary mapping package names to version strings (or None)
    """
    code_versions = {}

    # Get versions for core packages
    for code in _DEFAULT_VERSION_CODES:
        code_versions[code.__name__] = get_version_safe(code.__name__)

    if verbose:
        logger.info("Version Information:")
        for k, v in code_versions.items():
            logger.info(f"{k} {v}")

    return code_versions
