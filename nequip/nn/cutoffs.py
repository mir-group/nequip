import torch


class PolynomialCutoff(torch.nn.Module):
    _factor: float
    p: float

    def __init__(self, r_max: float, p: float = 6):
        r"""Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123

        Args:
            r_max (float): cutoff radius
            p (int)      : power used in envelope function
        """
        super().__init__()
        assert p >= 2.0
        self.p = float(p)
        self._factor = 1.0 / float(r_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate cutoff function.

        Args:
            x (torch.Tensor): input distance
        """
        x = x * self._factor
        out = 1.0
        out = out - (((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x, self.p))
        out = out + (self.p * (self.p + 2.0) * torch.pow(x, self.p + 1.0))
        out = out - ((self.p * (self.p + 1.0) / 2) * torch.pow(x, self.p + 2.0))
        return out * (x < 1.0)
