import torch
import os
import contextlib
from typing import Union, Final


def dtype_from_name(name: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    return {"float32": torch.float32, "float64": torch.float64}[name]


def dtype_to_name(name: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(name, str):
        return name
    return {torch.float32: "float32", torch.float64: "float64"}[name]


@contextlib.contextmanager
def torch_default_dtype(dtype):
    """Set `torch.get_default_dtype()` for the duration of a with block, cleaning up with a `finally`.

    Note that this is NOT thread safe, since `torch.set_default_dtype()` is not thread safe.
    """
    orig_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(orig_default_dtype)


# === floating point tolerances as env vars ===
_FLOAT64_MODEL_TOL: Final[float] = float(
    os.environ.get("NEQUIP_FLOAT64_MODEL_TOL", 1e-12)
)
_FLOAT32_MODEL_TOL: Final[float] = float(
    os.environ.get("NEQUIP_FLOAT32_MODEL_TOL", 5e-5)
)
_TF32_MODEL_TOL: Final[float] = float(os.environ.get("NEQUIP_TF32_MODEL_TOL", 1e-4))


def floating_point_tolerance(model_dtype: Union[str, torch.dtype]):
    """
    Consistent set of floating point tolerances for sanity checking based on ``model_dtype``, that also accounts for TF32 state.

    Assumes global dtype if ``float64``, and that TF32 will only ever be used if ``model_dtype`` is ``float32``.
    """
    using_tf32 = False
    if torch.cuda.is_available():
        # assume that both are set to be the same
        assert torch.backends.cuda.matmul.allow_tf32 == torch.backends.cudnn.allow_tf32
        using_tf32 = torch.torch.backends.cuda.matmul.allow_tf32
    return {
        torch.float32: _TF32_MODEL_TOL if using_tf32 else _FLOAT32_MODEL_TOL,
        "float32": _TF32_MODEL_TOL if using_tf32 else _FLOAT32_MODEL_TOL,
        torch.float64: _FLOAT64_MODEL_TOL,
        "float64": _FLOAT64_MODEL_TOL,
    }[model_dtype]
