import torch

import math

from e3nn.util.jit import compile_mode


@compile_mode("script")
class ShiftedSoftPlus(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x) - math.log(2.0)
