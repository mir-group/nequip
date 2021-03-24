"""Batched running statistics for PyTorch."""

from typing import Union, Tuple
from collections import Counter
import enum
import numbers

import torch


class Reduction(enum.Enum):
    MEAN = "mean"
    MEAN_STD = "mean_std"
    RMS = "rms"
    COUNT = "count"


# TODO: impliment counting
# TODO: impliment stds
class RunningStats:
    """Compute running statistics over batches of samples.

    Args:
        dim (int or tuple of int): the shape of a sample
        reduction (Reduction): the statistic to compute
    """

    def __init__(
        self, dim: Union[int, Tuple[int]] = 1, reduction: Reduction = Reduction.MEAN
    ):
        if isinstance(dim, numbers.Integral):
            self._dim = (dim,)
        elif isinstance(dim, tuple):
            self._dim = dim
        else:
            raise TypeError(f"Invalid dim {dim}")

        if reduction not in (Reduction.MEAN, Reduction.RMS):
            raise NotImplementedError(f"Reduction {reduction} not yet implimented")
        self._reduction = reduction
        self.reset()

    def accumulate_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Accumulate a batch of samples into running statistics.

        Args:
            batch (torch.Tensor [N_samples, dim]): the batch of samples to process.

        Returns:
            the aggregated statistics _for this batch_. Accumulated statistics up to this point can be retreived with ``current_result()``.
        """
        assert batch.shape[1:] == self._dim
        N = batch.shape[0]
        if self._reduction == Reduction.MEAN:
            new = batch.sum(dim=0)
            self._state += (new - N * self._state) / (self._n + N)
            self._n += N
            # for the batch
            new /= N
            return new
        elif self._reduction == Reduction.RMS:
            new = torch.square(batch).sum(dim=0)
            self._state += (new - N * self._state) / (self._n + N)
            self._n += N
            # for the batch
            new /= N
            return new.sqrt()

    def reset(self) -> None:
        """Forget all previously accumulated state."""
        self._n = 0
        if hasattr(self, "_state"):
            self._state.fill_(0.0)
        else:
            self._state = torch.zeros(self._dim)

    def current_result(self):
        """Get the current value of the running statistc."""
        if self._reduction == Reduction.MEAN:
            return self._state.clone()
        elif self._reduction == Reduction.RMS:
            return torch.sqrt(self._state)

    @property
    def n(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def reduction(self) -> Reduction:
        return self._reduction
