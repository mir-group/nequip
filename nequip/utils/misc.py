from typing import Union
import torch


def dtype_from_name(name: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    return {"float32": torch.float32, "float64": torch.float64}[name]
