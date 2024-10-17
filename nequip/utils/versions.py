import packaging.version

import torch
import e3nn
import nequip

from ..__init__ import _DISCOVERED_NEQUIP_EXTENSION
from .logger import RankedLogger
from typing import Tuple, Final
import importlib

logger = RankedLogger(__name__, rank_zero_only=True)

_TORCH_IS_GE_1_13: Final[bool] = packaging.version.parse(
    torch.__version__
) >= packaging.version.parse("1.13.0")

# get versions of torch, e3nn, nequip and all extension packages
_DEFAULT_VERSION_CODES = [torch, e3nn, nequip]
for ep in _DISCOVERED_NEQUIP_EXTENSION:
    _DEFAULT_VERSION_CODES.append(importlib.import_module(ep.value))


def get_current_code_versions() -> Tuple[dict, dict]:
    code_versions = {}
    for code in _DEFAULT_VERSION_CODES:
        code_versions[code.__name__] = str(code.__version__)

    logger.info("{:^29}".format("Version Information"))
    for k, v in code_versions.items():
        logger.info(f"{k:^14}:{v:^14}")

    return code_versions
