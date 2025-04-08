# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from torchmetrics import Metric
from nequip.utils.global_dtype import _GLOBAL_DTYPE
from typing import Callable


class _MeanX(Metric):
    """Hidden base class for mean statistics.

    Computes running means with the mean part of Welford's one-pass algorithm for running variance. Can be subclassed for other types of mean statistics, e.g. MeanAbsolute, MeanSquare, etc with ``modifier`` argument.
    """

    def __init__(self, modifier: Callable = torch.nn.Identity(), **kwargs):
        super().__init__(**kwargs)
        self.modifier = modifier
        # use the running mean part of Welford's one-pass algorithm for running variance
        # but keep sum and count as variables to be updated correctly during distributed training
        # reasoning: we avoid accumulated sum (big number) += new sum (small number) during each update so it should be more numerically stable, but we sync sums across devices assuming that the sum on each device is of the same order of magnitude
        self.add_state("sum", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, data: torch.Tensor) -> None:
        # short circuit if data tensor is empty
        sample_count = data.numel()
        if sample_count > 0:
            data = data.to(_GLOBAL_DTYPE)
            # subtract means instead of add sums to reduce precision loss
            current_mean = (
                self.sum.div(self.count) if torch.is_nonzero(self.count) else 0
            )
            sample_mean = self.modifier(data).mean()
            delta = sample_mean - current_mean
            new_count = self.count + sample_count
            new_mean = current_mean + delta * sample_count / new_count
            # update count and sum
            self.count = new_count
            self.sum = new_mean * self.count

    def compute(self) -> torch.Tensor:
        return self.sum.div(self.count)


class Mean(_MeanX):
    """Mean computed in a running fashion."""

    def __init__(self, **kwargs):
        super().__init__(modifier=torch.nn.Identity(), **kwargs)

    def __str__(self) -> str:
        return "mean"


class MeanAbsolute(_MeanX):
    """Mean of absolute value computed in a running fashion."""

    def __init__(self, **kwargs):
        super().__init__(modifier=torch.abs, **kwargs)

    def __str__(self) -> str:
        return "mean_abs"


class RootMeanSquare(_MeanX):
    """Root mean square computed in a running fashion."""

    def __init__(self, **kwargs):
        super().__init__(modifier=torch.square, **kwargs)

    def compute(self) -> torch.Tensor:
        """"""
        mean_square = super().compute()
        return torch.sqrt(mean_square)

    def __str__(self) -> str:
        return "rms"


class StandardDeviation(Metric):
    """Standard deviation computed in a running fashion with Welford's online algorithm.

    Args:
        squared (bool): if ``True``, returns variance, else returns standard deviation
        unbiased (bool): whether to use the unbiased estimate for standard deviation
    """

    def __init__(self, squared: bool = False, unbiased=True, **kwargs):
        super().__init__(**kwargs)
        self.squared = squared
        self.unbiased = unbiased
        # TODO: implement the correct dist_reduce_fx for distributed use
        self.add_state("M2", default=torch.tensor(0))
        self.add_state("mean", default=torch.tensor(0))
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, data: torch.Tensor) -> None:
        """"""
        # short circuit if data tensor is empty
        sample_count = data.numel()
        if sample_count > 0:
            data = data.to(_GLOBAL_DTYPE)
            # compute sample stats
            sample_mean = data.mean()
            sample_M2 = (data - sample_mean).square().sum()
            # auxiliary variables
            delta = sample_mean - self.mean
            new_count = self.count + sample_count
            mean_change = delta * sample_count / new_count
            # update
            self.mean = self.mean + mean_change
            self.M2 = self.M2 + sample_M2 + delta * mean_change * self.count
            self.count = new_count

    def compute(self) -> torch.Tensor:
        """"""
        denom = self.count - 1 if self.unbiased else self.count
        variance = self.M2.div(denom)
        return variance if self.squared else torch.sqrt(variance)

    def __str__(self) -> str:
        return "var" if self.squared else "std"


class Max(Metric):
    """Largest entry seen.

    Args:
        abs (bool): whether to use absolute values
    """

    def __init__(self, abs: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.abs = abs
        self.add_state("max", default=torch.tensor(-float("inf")), dist_reduce_fx="max")

    def update(self, data: torch.Tensor) -> None:
        """"""
        if data.numel() > 0:
            self.max = torch.maximum(
                self.max, data.abs().max() if self.abs else data.max()
            )

    def compute(self) -> torch.Tensor:
        """"""
        return self.max

    def __str__(self) -> str:
        return "absmax" if self.abs else "max"


class Min(Metric):
    """Smallest entry seen.

    Args:
        abs (bool): whether to use absolute values
    """

    def __init__(self, abs: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.abs = abs
        self.add_state("min", default=torch.tensor(float("inf")), dist_reduce_fx="min")

    def update(self, data: torch.Tensor) -> None:
        """"""
        if data.numel() > 0:
            self.min = torch.minimum(
                self.min, data.abs().min() if self.abs else data.min()
            )

    def compute(self) -> torch.Tensor:
        """"""
        return self.min

    def __str__(self) -> str:
        return "absmin" if self.abs else "min"


class Count(Metric):
    """Total number of entries."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, data: torch.Tensor) -> None:
        """"""
        self.count += data.numel()

    def compute(self) -> torch.Tensor:
        """"""
        return self.count

    def __str__(self) -> str:
        return "count"
