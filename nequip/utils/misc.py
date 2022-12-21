from typing import Union
import contextlib

import torch


def dtype_from_name(name: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    return {"float32": torch.float32, "float64": torch.float64}[name]


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
