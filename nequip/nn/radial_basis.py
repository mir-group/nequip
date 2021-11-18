from typing import Optional
import math

import torch

from torch import nn

from e3nn.math import soft_one_hot_linspace
from e3nn.util.jit import compile_mode


@compile_mode("trace")
class e3nn_basis(nn.Module):
    r_max: float
    r_min: float
    e3nn_basis_name: str
    num_basis: int

    def __init__(
        self,
        r_max: float,
        r_min: Optional[float] = None,
        e3nn_basis_name: str = "gaussian",
        num_basis: int = 8,
    ):
        super().__init__()
        self.r_max = r_max
        self.r_min = r_min if r_min is not None else 0.0
        self.e3nn_basis_name = e3nn_basis_name
        self.num_basis = num_basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return soft_one_hot_linspace(
            x,
            start=self.r_min,
            end=self.r_max,
            number=self.num_basis,
            basis=self.e3nn_basis_name,
            cutoff=True,
        )

    def _make_tracing_inputs(self, n: int):
        return [{"forward": (torch.randn(5, 1),)} for _ in range(n)]


class BesselBasis(nn.Module):
    r_max: float
    prefactor: float

    def __init__(self, r_max, num_basis=8, trainable=True):
        r"""Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)

        return self.prefactor * (numerator / x.unsqueeze(-1))


# class GaussianBasis(nn.Module):
#     r_max: float

#     def __init__(self, r_max, r_min=0.0, num_basis=8, trainable=True):
#         super().__init__()

#         self.trainable = trainable
#         self.num_basis = num_basis

#         self.r_max = float(r_max)
#         self.r_min = float(r_min)

#         means = torch.linspace(self.r_min, self.r_max, self.num_basis)
#         stds = torch.full(size=means.size, fill_value=means[1] - means[0])
#         if self.trainable:
#             self.means = nn.Parameter(means)
#             self.stds = nn.Parameter(stds)
#         else:
#             self.register_buffer("means", means)
#             self.register_buffer("stds", stds)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = (x[..., None] - self.means) / self.stds
#         x = x.square().mul(-0.5).exp() / self.stds  # sqrt(2 * pi)
