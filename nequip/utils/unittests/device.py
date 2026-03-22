# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import List

import torch


def get_accelerators() -> List[str]:
    """Return available accelerator device types in priority order."""
    accelerators: List[str] = []
    if torch.cuda.is_available():
        accelerators.append("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        accelerators.append("xpu")
    return accelerators


def get_test_devices(*, include_cpu: bool = True) -> List[str]:
    """Return test devices, optionally including CPU first."""
    devices = list(get_accelerators())
    if include_cpu:
        devices.insert(0, "cpu")
    return devices
