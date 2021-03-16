import torch
from torch import nn


class PolynomialCutoff(nn.Module):
    def __init__(self, r_max=5.0, p=6):
        r"""Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        p : int
            Power used in envelope function
        """
        super(PolynomialCutoff, self).__init__()

        self.register_buffer("p", torch.Tensor([p]))
        self.register_buffer("r_max", torch.Tensor([r_max]))

    def forward(self, x):
        """
        Evaluate cutoff function.

        x: torch.Tensor, input distance
        """
        envelope = (
            1.0
            - ((self.p + 1.0) * (self.p + 2.0) / 2.0)
            * torch.pow(x / self.r_max, self.p)
            + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1.0)
            - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2.0)
        )

        envelope *= (x < self.r_max).float()
        return envelope
