import torch
import numpy as np

from torch import nn


class BesselBasis(nn.Module):
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

        self.register_buffer("r_max", torch.Tensor([r_max]))
        self.register_buffer("prefactor", torch.Tensor([2.0 / self.r_max]))

        self.bessel_weights = torch.linspace(
            start=1.0, end=num_basis, steps=num_basis
        ) * torch.Tensor([np.pi])

        if self.trainable:
            self.bessel_weights = nn.Parameter(self.bessel_weights)

    def forward(self, x):
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)

        return self.prefactor * (numerator / x.unsqueeze(-1))
