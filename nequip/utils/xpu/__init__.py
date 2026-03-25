from .xpu import XPUAccelerator

__all__ = ["XPUAccelerator"]

import sys
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.accelerators import AcceleratorRegistry
from lightning.fabric.utilities.registry import _register_classes


_register_classes(
    AcceleratorRegistry, "register_accelerators", sys.modules[__name__], Accelerator
)
