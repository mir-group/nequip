import torch


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]
