import warnings
from packaging import version
import os

import torch

import e3nn
import e3nn.util.jit

from .misc import dtype_from_name
from .test import set_irreps_debug
from .config import Config


# for multiprocessing, we need to keep track of our latest global options so
# that we can reload/reset them in worker processes. While we could be more careful here,
# to keep only relevant keys, configs should have only small values (no big objects)
# and those should have references elsewhere anyway, so keeping references here is fine.
_latest_global_config = {}


def _get_latest_global_options() -> dict:
    """Get the config used latest to ``_set_global_options``.

    This is useful for getting worker processes into the same state as the parent.
    """
    global _latest_global_config
    return _latest_global_config


def _set_global_options(config, warn_on_override: bool = False) -> None:
    """Configure global options of libraries like `torch` and `e3nn` based on `config`.

    Args:
        warn_on_override: if True, will try to warn if new options are inconsistant with previously set ones.
    """
    # update these options into the latest global config.
    global _latest_global_config
    _latest_global_config.update(Config.as_dict(config))
    # Set TF32 support
    # See https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available() and "allow_tf32" in config:
        if torch.torch.backends.cuda.matmul.allow_tf32 is not config["allow_tf32"]:
            # update the setting
            if warn_on_override:
                warnings.warn(
                    f"Setting the GLOBAL value for allow_tf32 to {config['allow_tf32']} which is different than the previous value of {torch.torch.backends.cuda.matmul.allow_tf32}"
                )
            torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]
            torch.backends.cudnn.allow_tf32 = config["allow_tf32"]

    # Temporary warning due to unresolved upstream issue
    torch_version = version.parse(torch.__version__)

    if torch_version >= version.parse("1.11"):
        # PyTorch >= 1.11
        k = "_jit_fusion_strategy"
        if k in config:
            new_strat = config.get(k)
            old_strat = torch.jit.set_fusion_strategy(new_strat)
            if warn_on_override and old_strat != new_strat:
                warnings.warn(
                    f"Setting the GLOBAL value for jit fusion strategy to `{new_strat}` which is different than the previous value of `{old_strat}`"
                )
    else:
        # For avoiding 20 steps of painfully slow JIT recompilation
        # See https://github.com/pytorch/pytorch/issues/52286
        k = "_jit_bailout_depth"
        if k in config:
            new_depth = config[k]
            old_depth = torch._C._jit_set_bailout_depth(new_depth)
            if warn_on_override and old_depth != new_depth:
                warnings.warn(
                    f"Setting the GLOBAL value for jit bailout depth to `{new_depth}` which is different than the previous value of `{old_depth}`"
                )

    # Deal with fusers
    # The default PyTorch fuser changed to nvFuser in 1.12
    # fuser1 is NNC, fuser2 is nvFuser
    # See https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#fusers
    # And https://github.com/pytorch/pytorch/blob/e0a0f37a11164f59b42bc80a6f95b54f722d47ce/torch/jit/_fuser.py#L46
    # Also https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/codegen/cuda/README.md
    # Also https://github.com/pytorch/pytorch/blob/66fb83293e6a6f527d3fde632e3547fda20becea/torch/csrc/jit/OVERVIEW.md?plain=1#L1201
    # https://github.com/search?q=repo%3Apytorch%2Fpytorch%20PYTORCH_JIT_USE_NNC_NOT_NVFUSER&type=code
    # We follow the approach they have explicitly built for disabling nvFuser in favor of NNC:
    # https://github.com/pytorch/pytorch/blob/66fb83293e6a6f527d3fde632e3547fda20becea/torch/csrc/jit/codegen/cuda/README.md?plain=1#L214
    #
    #     There are three ways to disable nvfuser. Listed below with descending priorities:
    #      - Force using NNC instead of nvfuser for GPU fusion with env variable `export PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1`.
    #      - Disabling nvfuser with torch API `torch._C._jit_set_nvfuser_enabled(False)`.
    #      - Disable nvfuser with env variable `export PYTORCH_JIT_ENABLE_NVFUSER=0`.
    #
    k = "PYTORCH_JIT_USE_NNC_NOT_NVFUSER"
    if k in os.environ and int(os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"]) == 0:
        warnings.warn(
            "Do NOT manually set PYTORCH_JIT_USE_NNC_NOT_NVFUSER=0 unless you know exactly what you're doing!"
        )
    else:
        os.environ[k] = "1"

    # TODO: warn_on_override for the rest here?
    if config.get("model_debug_mode", False):
        set_irreps_debug(enabled=True)
    if "default_dtype" in config:
        old_dtype = torch.get_default_dtype()
        new_dtype = dtype_from_name(config["default_dtype"])
        if warn_on_override and old_dtype != new_dtype:
            warnings.warn(
                f"Setting the GLOBAL value for torch.set_default_dtype to `{new_dtype}` which is different than the previous value of `{old_dtype}`"
            )
        torch.set_default_dtype(new_dtype)
    if config.get("grad_anomaly_mode", False):
        torch.autograd.set_detect_anomaly(True)

    e3nn.set_optimization_defaults(**config.get("e3nn_optimization_defaults", {}))

    return
