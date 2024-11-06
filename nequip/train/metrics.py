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


class HuberLoss(_MeanX):
    """Huber loss (see `torch.nn.HuberLoss <https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html>`_)"""

    def __init__(self, reduction="mean", delta=1.0, **kwargs):

        assert reduction in ["mean", "sum"]

        def _huber(x):
            absx = torch.abs(x)
            return torch.where(
                absx < delta, 0.5 * x.square(), delta * (absx - 0.5 * delta)
            )

        super().__init__(modifier=_huber, **kwargs)
        self.reduction = reduction

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """"""
        super().update(preds - target)

    def compute(self) -> torch.Tensor:
        """"""
        if self.reduction == "mean":
            return self.sum.div(self.count)
        elif self.reduction == "sum":
            return self.sum

    def __str__(self) -> str:
        return "huber"
