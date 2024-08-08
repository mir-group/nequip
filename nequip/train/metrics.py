import torch
from nequip.data.stats import _MeanX


class MeanAbsoluteError(_MeanX):
    """Mean absolute error."""

    def __init__(self, **kwargs):
        super().__init__(modifier=torch.abs, **kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """"""
        super().update(preds - target)

    def __str__(self) -> str:
        return "mae"


class MeanSquaredError(_MeanX):
    """Mean squared error."""

    def __init__(self, **kwargs):
        super().__init__(modifier=torch.square, **kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """"""
        super().update(preds - target)

    def __str__(self) -> str:
        return "mse"


class RootMeanSquaredError(MeanSquaredError):
    """Root mean squared error."""

    def compute(self) -> torch.Tensor:
        """"""
        return torch.sqrt(self.sum.div(self.count))

    def __str__(self) -> str:
        return "rmse"
