# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch


class PolynomialCutoff(torch.nn.Module):

    def __init__(self, p: float = 6):
        r"""Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123

        Args:
            r_max (float): cutoff radius
            p (int)      : power used in envelope function
        """
        super().__init__()
        assert p >= 2.0
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate cutoff function.

        Args:
            x (torch.Tensor): input distance
        """
        out = 1.0
        out = out - (((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x, self.p))
        out = out + (self.p * (self.p + 2.0) * torch.pow(x, self.p + 1.0))
        out = out - ((self.p * (self.p + 1.0) / 2) * torch.pow(x, self.p + 2.0))
        return out * (x < 1.0)
