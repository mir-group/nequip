from typing import Union, List
import contextlib

import torch


def dtype_from_name(name: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    return {"float32": torch.float32, "float64": torch.float64}[name]


def dtype_to_name(name: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(name, str):
        return name
    return {torch.float32: "float32", torch.float64: "float64"}[name]


def get_default_device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def format_type_vals(
    vals: List[float], type_names: List[str], element_formatter: str = ".6f"
) -> str:

    if vals is None:
        return f"[{', '.join(type_names)}: None]"

    if len(vals) == 1:
        return (f"[{', '.join(type_names)}: {{:{element_formatter}}}]").format(vals[0])
    elif len(vals) == len(type_names):
        return (
            "["
            + ", ".join(
                f"{{{i}[0]}}: {{{i}[1]:{element_formatter}}}" for i in range(len(vals))
            )
            + "]"
        ).format(*zip(type_names, vals))
    else:
        raise ValueError(
            f"Don't know how to format vals=`{vals}` for types {type_names} with element_formatter=`{element_formatter}`"
        )
