# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from lightning.pytorch import seed_everything

import e3nn

from .global_dtype import _GLOBAL_DTYPE

import warnings
import os
from typing import List, Dict, Tuple, Union, Final

from .resolvers import _register_resolvers

# for multiprocessing, we need to keep track of our latest global state so
# that we can reload/reset them in worker processes. While we could be more careful here,
# to keep only relevant keys, configs should have only small values (no big objects)
# and those should have references elsewhere anyway, so keeping references here is fine.
_latest_global_config = {}

_GLOBAL_STATE_INITIALIZED = False

_MULTIPROCESSING_SHARING_STRATEGY: Final[str] = os.environ.get(
    "NEQUIP_MULTIPROCESSING_SHARING_STRATEGY", "file_system"
)

# === hardcode jit fusion strategy ===
# avoid 20 iters of pain, see https://github.com/pytorch/pytorch/issues/52286
# Quote from eelison in PyTorch slack:
# https://pytorch.slack.com/archives/CDZD1FANA/p1644259272007529?thread_ts=1644064449.039479&cid=CDZD1FANA
# > Right now the default behavior is to specialize twice on static shapes and then on dynamic shapes.
# > To reduce warmup time you can do something like setFusionStrartegy({{FusionBehavior::DYNAMIC, 3}})
# > ... Although we would wouldn't really expect to recompile a dynamic shape fusion in a model,
# > provided broadcasting patterns remain fixed
# We default to DYNAMIC alone because the number of edges is always dynamic,
# even if the number of atoms is fixed
_JIT_FUSION_STRATEGY: Final[List[Tuple[Union[str, int]]]] = [("DYNAMIC", 10)]

# === global state metadata ===
# the global state that we want to track in a packaged model
TF32_KEY: Final[str] = "allow_tf32"

# only TF32 for now, but future-proofs for "singular source of truth" wrt metadata keys related to global state
GLOBAL_STATE_METADATA_KEYS: Final[List[str]] = [TF32_KEY]


def get_latest_global_state(only_metadata_related=False) -> Dict:
    """Queries the latest (exposed) global state."""
    global _latest_global_config
    if only_metadata_related:
        return {
            k: v
            for k, v in _latest_global_config.items()
            if k in GLOBAL_STATE_METADATA_KEYS
        }
    else:
        return _latest_global_config


def global_state_initialized():
    global _GLOBAL_STATE_INITIALIZED
    return _GLOBAL_STATE_INITIALIZED


def set_global_state(
    allow_tf32: bool = False,
    warn_on_override: bool = False,
) -> None:
    """Configure global state.

    Args:
        warn_on_override: whether to warn if new state is inconsistent with previously set ones
    """
    # === global state initialization ===
    global _GLOBAL_STATE_INITIALIZED
    if not _GLOBAL_STATE_INITIALIZED:
        # === set global seed ===
        seed_everything(123, workers=True, verbose=False)

        # === register default omegaconf resolvers ===
        _register_resolvers()

        # === set global dtype ===
        torch.set_default_dtype(_GLOBAL_DTYPE)

        # this applies for torch >= 1.11 (our minimum torch version satisfies this)
        # since we hardcode it, there's no need to warn
        _ = torch.jit.set_fusion_strategy(_JIT_FUSION_STRATEGY)

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

        # === e3nn optimization flags ===
        # we initialize them all to true
        e3nn.set_optimization_defaults(
            specialized_code=True,
            optimize_einsums=True,
            jit_script_fx=True,
        )

        # ENVIRONMENT VARIABLES
        # torch.multiprocessing fix for batch_size=1
        # see https://stackoverflow.com/questions/48250053/pytorchs-dataloader-too-many-open-files-error-when-no-files-should-be-open
        torch.multiprocessing.set_sharing_strategy(_MULTIPROCESSING_SHARING_STRATEGY)

        # === initialize TF32 to False ===
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        _GLOBAL_STATE_INITIALIZED = True

    # === TF32 ===
    # See https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # NOTE: it is also possible to use https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if torch.cuda.is_available():
        if torch.torch.backends.cuda.matmul.allow_tf32 is not allow_tf32:
            # update the setting
            if warn_on_override:
                warnings.warn(
                    f"Setting the GLOBAL value for allow_tf32 to {allow_tf32} which is different than the previous value of {torch.torch.backends.cuda.matmul.allow_tf32}"
                )
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32

    # === update global config ===
    # NOTE: fixed fields are not reflected in the global config
    global _latest_global_config
    _latest_global_config.update(
        {
            TF32_KEY: allow_tf32,
        }
    )
    return
