import torch

import math

from e3nn.util.jit import compile_mode


@compile_mode("script")
class ShiftedSoftPlus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftPlus, self).__init__()
        self.func = torch.nn.Softplus()

    def forward(self, x: torch.Tensor):
        return self.func(x) - math.log(2.0)
