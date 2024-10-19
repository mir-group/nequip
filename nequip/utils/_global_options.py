import torch

from lightning.pytorch import seed_everything

import e3nn
import e3nn.util.jit

from .misc import dtype_to_name

import warnings
import os
from typing import List, Tuple, Union, Final, Optional


# for multiprocessing, we need to keep track of our latest global options so
# that we can reload/reset them in worker processes. While we could be more careful here,
# to keep only relevant keys, configs should have only small values (no big objects)
# and those should have references elsewhere anyway, so keeping references here is fine.
_latest_global_config = {}

# singular source of global dtype that dictates the dtype of the data
# which is always float64
_GLOBAL_DTYPE = torch.float64


_MULTIPROCESSING_SHARING_STRATEGY: Final[str] = os.environ.get(
    "NEQUIP_MULTIPROCESSING_SHARING_STRATEGY", "file_system"
)


def _get_latest_global_options() -> dict:
    """Get the config used latest to ``_set_global_options``.

    This is useful for getting worker processes into the same state as the parent.
    """
    global _latest_global_config
    return _latest_global_config


def _set_global_options(
    seed: Optional[int] = None,
    # avoid 20 iters of pain, see https://github.com/pytorch/pytorch/issues/52286
    # Quote from eelison in PyTorch slack:
    # https://pytorch.slack.com/archives/CDZD1FANA/p1644259272007529?thread_ts=1644064449.039479&cid=CDZD1FANA
    # > Right now the default behavior is to specialize twice on static shapes and then on dynamic shapes.
    # > To reduce warmup time you can do something like setFusionStrartegy({{FusionBehavior::DYNAMIC, 3}})
    # > ... Although we would wouldn't really expect to recompile a dynamic shape fusion in a model,
    # > provided broadcasting patterns remain fixed
    # We default to DYNAMIC alone because the number of edges is always dynamic,
    # even if the number of atoms is fixed:
    _jit_fusion_strategy: List[Tuple[Union[str, int]]] = [("DYNAMIC", 3)],
    # Due to what appear to be ongoing bugs with nvFuser, we default to NNC (fuser1) for now:
    # TODO: still default to NNC on CPU regardless even if change this for GPU
    # TODO: default for ROCm?
    # _jit_fuser="fuser1",  # TODO: what is this?
    allow_tf32: bool = False,
    # e3nn_optimization_defaults
    specialized_code: bool = True,
    optimize_einsums: bool = True,
    jit_script_fx: bool = True,
    warn_on_override: bool = False,
) -> None:
    """Configure global options of libraries like `torch` and `e3nn` based on `config`.

    Args:
        warn_on_override: if True, will try to warn if new options are inconsistant with previously set ones.
    """
    # update these options into the latest global config.
    global _latest_global_config
    _latest_global_config.update(
        {
            "seed": seed,
            "_jit_fusion_strategy": _jit_fusion_strategy,
            "allow_tf32": allow_tf32,
            "specialized_code": specialized_code,
            "optimize_einsums": optimize_einsums,
            "jit_script_fx": jit_script_fx,
            "warn_on_override": warn_on_override,
            "default_dtype": dtype_to_name(_GLOBAL_DTYPE),
        }
    )

    # set global seed
    if seed is not None:
        seed_everything(seed, workers=True)

    # Set TF32 support
    # See https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available():
        if torch.torch.backends.cuda.matmul.allow_tf32 is not allow_tf32:
            # update the setting
            if warn_on_override:
                warnings.warn(
                    f"Setting the GLOBAL value for allow_tf32 to {allow_tf32} which is different than the previous value of {torch.torch.backends.cuda.matmul.allow_tf32}"
                )
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32

    # this applies for torch >= 1.11 (our minimum torch version satisfies this)
    new_strat = _jit_fusion_strategy
    old_strat = torch.jit.set_fusion_strategy(new_strat)
    if warn_on_override and old_strat != new_strat:
        warnings.warn(
            f"Setting the GLOBAL value for jit fusion strategy to `{new_strat}` which is different than the previous value of `{old_strat}`"
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
    if k in os.environ:
        if os.environ[k] != "1":
            warnings.warn(
                "Do NOT manually set PYTORCH_JIT_USE_NNC_NOT_NVFUSER=0 unless you know exactly what you're doing!"
            )
    else:
        os.environ[k] = "1"

    torch.set_default_dtype(_GLOBAL_DTYPE)

    e3nn.set_optimization_defaults(
        specialized_code=specialized_code,
        optimize_einsums=optimize_einsums,
        jit_script_fx=jit_script_fx,
    )

    # ENVIRONMENT VARIABLES
    # torch.multiprocessing fix for batch_size=1
    # see https://stackoverflow.com/questions/48250053/pytorchs-dataloader-too-many-open-files-error-when-no-files-should-be-open
    torch.multiprocessing.set_sharing_strategy(_MULTIPROCESSING_SHARING_STRATEGY)

    return
