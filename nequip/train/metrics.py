# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
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
    """Huber loss (see `torch.nn.HuberLoss <https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html>`_)

    Note that ``delta`` takes on the units of the target and prediction tensors.
    """

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


class StratifiedHuberForceLoss(_MeanX):
    """Stratified Huber loss for vectors (forces)
    (see `torch.nn.HuberLoss <https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html>`_).

    This metrics class implements a stratified/conditional Huber loss, where the Huber ``delta`` parameter is scaled based on the magnitude of the reference vector (i.e. force), by providing a ``delta_dict`` of ``{lower bound: delta parameter}`` where the loss contributions for all vectors with a magnitude between lower bound and the next lower bound are computed as a Huber loss with the corresponding ``delta`` parameter.

    Note that ``delta`` values take on the units of the target and prediction tensors.

    If the first lower bound in ``delta_dict`` is not 0 (typically recommended),
    then a MSELoss (divided by 2; matching Huber loss in the L2 regime (``|x| < delta``),
    see `torch.nn.HuberLoss <https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html>`_) is used for vectors with a magnitude smaller than the first lower bound.
    """

    def __init__(self, delta_dict, reduction="mean", **kwargs):
        if min(delta_dict.keys()) > 0:
            # add a 0 - lower-bound but with infinite delta so always in the L2 regime
            delta_dict = {0: float("inf"), **delta_dict}

        self.delta_dict = delta_dict  # dict of lower bound: delta parameter
        assert reduction in ["mean", "sum"]
        assert (
            len(self.delta_dict) >= 2
        ), "At least two delta values are required, otherwise use standard HuberLoss instead."

        super().__init__(**kwargs)
        self.reduction = reduction

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """"""
        # templated from the `conditional_huber_forces` function from MACE:
        bounds = list(self.delta_dict.keys())
        deltas = list(self.delta_dict.values())

        stratified_losses = torch.zeros_like(preds)
        vector_magnitudes = torch.norm(target, dim=-1)
        bounds_masks = [vector_magnitudes >= bounds[i] for i in range(len(bounds))]
        for i in range(len(bounds)):
            stratum_mask = bounds_masks[i] & (
                ~bounds_masks[i + 1] if (i + 1) < len(bounds) else True
            )
            stratified_losses[stratum_mask] = torch.nn.functional.huber_loss(
                target[stratum_mask],
                preds[stratum_mask],
                reduction="none",
                delta=deltas[i],
            )

        super().update(stratified_losses)

    def compute(self) -> torch.Tensor:
        """"""
        if self.reduction == "mean":
            return self.sum.div(self.count)
        elif self.reduction == "sum":
            return self.sum

    def __str__(self) -> str:
        return "stratified huber"
