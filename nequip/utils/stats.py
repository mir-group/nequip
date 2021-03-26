"""Batched running statistics for PyTorch."""

from typing import Union, Tuple, Optional
from collections import Counter
import enum
import numbers

import torch
from torch_scatter import scatter


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
        self,
        dim: Union[int, Tuple[int]] = 1,
        reduction: Reduction = Reduction.MEAN,
        weighted: bool = False,
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
        self.weighted = weighted
        if weighted:
            raise NotImplementedError
        self.reset()

    def accumulate_batch(
        self, batch: torch.Tensor, accumulate_by: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Accumulate a batch of samples into running statistics.

        Args:
            batch (torch.Tensor [N_samples, dim]): the batch of samples to process.

        Returns:
            the aggregated statistics _for this batch_. Accumulated statistics up to this point can be retreived with ``current_result()``.
        """
        assert batch.shape[1:] == self._dim

        if self._reduction == Reduction.COUNT:
            raise NotImplementedError
        else:
            if accumulate_by is None:
                # accumulate everything into the first bin
                N = batch.shape[0]
                if self._reduction == Reduction.MEAN:
                    new_sum = batch.sum(dim=0)
                elif self._reduction == Reduction.RMS:
                    new_sum = torch.square(batch).sum(dim=0)

                # accumulate
                self._state[0:1] += (new_sum - N * self._state) / (self._n[0:1] + N)
                self._n[0:1] += N

                # for the batch
                new_sum /= N

                if self._reduction == Reduction.RMS:
                    new_sum.sqrt_()

                return new_sum
            else:
                if self._reduction == Reduction.MEAN:
                    new_sum = scatter(batch, accumulate_by, dim=0)
                elif self._reduction == Reduction.RMS:
                    new_sum = scatter(torch.square(batch), accumulate_by, dim=0)

                if new_sum.shape[0] > self._n_bins:
                    # time to expand
                    N_to_add = new_sum.shape[0] - self._n_bins
                    self._state = torch.cat(
                        (self._state, self._state.new_zeros((N_to_add,) + self._dim)),
                        dim=0,
                    )
                    self._n = torch.cat((self._n, self._n.new_zeros((N_to_add,)+tuple(1 for i in self._dim))), dim=0)
                    assert len(self._state) == self._n_bins + N_to_add

                N = torch.bincount(accumulate_by).reshape([-1, 1])

                N_bins_new = new_sum.shape[0]

                self._state[:N_bins_new] += (new_sum - N * self._state[:N_bins_new]) / (
                    self._n[:N_bins_new] + N
                )
                self._n[:N_bins_new] += N

                new_sum /= N

                if self._reduction == Reduction.RMS:
                    new_sum.sqrt_()

                return new_sum

    def reset(self) -> None:
        """Forget all previously accumulated state."""
        if hasattr(self, "_state"):
            self._state.fill_(0.0)
            self._n.fill_(0)
        else:
            self._n_bins = 1
            self._n = torch.zeros((self._n_bins,)+tuple(1 for i in self._dim), dtype=torch.long)
            self._state = torch.zeros((self._n_bins,) + self._dim)

    def current_result(self):
        """Get the current value of the running statistc."""
        if self._reduction == Reduction.MEAN:
            return self._state.clone()
        elif self._reduction == Reduction.RMS:
            return torch.sqrt(self._state)

    @property
    def n(self) -> torch.Tensor:
        return self._n.squeeze(0)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def reduction(self) -> Reduction:
        return self._reduction
