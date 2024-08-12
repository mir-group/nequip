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
    data: list, type_names: List[str], element_formatter: str = ".6f"
) -> str:
    data = torch.as_tensor(data) if data is not None else None

    if data.numel() == 1:
        data = torch.tensor(data.item())

    if data is None:
        return f"[{', '.join(type_names)}: None]"
    elif data.ndim == 0:
        return (f"[{', '.join(type_names)}: {{:{element_formatter}}}]").format(data)
    elif data.ndim == 1 and len(data) == len(type_names):
        return (
            "["
            + ", ".join(
                f"{{{i}[0]}}: {{{i}[1]:{element_formatter}}}" for i in range(len(data))
            )
            + "]"
        ).format(*zip(type_names, data))
    else:
        raise ValueError(
            f"Don't know how to format data=`{data}` for types {type_names} with element_formatter=`{element_formatter}`"
        )
