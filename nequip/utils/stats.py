"""Batched running statistics for PyTorch."""

from typing import Union, Tuple, Optional
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
        weight: str =False
    ):
        if isinstance(dim, numbers.Integral):
            self._dim = (dim,)
        elif isinstance(dim, tuple):
            self._dim = dim
        else:
            raise TypeError(f"Invalid dim {dim}")

        if reduction not in (Reduction.MEAN, Reduction.RMS):
            raise NotImplementedError(f"Reduction {reduction} not yet implimented")
        self.weight = weight
        self._reduction = reduction
        self.reset()

    def accumulate_batch(self, batch: torch.Tensor, weights:Optional[torch.Tensor]=None) -> torch.Tensor:
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
        elif self._reduction == Reduction.RMS:
            new = torch.square(batch).sum(dim=0)
        
        weight = weights.sum(dim=0) if self.weight else None

        return self.accumulate_sum(new_sum=new, N=N, weight=weight)




        species_index = ref[AtomicDataDict.SPECIES_INDEX_KEY]
        _, inverse_species_index, counts = torch.unique(species_index, return_inverse=True, return_counts=True)

        if atomic_weight_on:
            # TO DO
            per_species_weight = scatter(weights, inverse_species_index, dim=0)
            per_species_loss = scatter(per_atom_loss, inverse_species_index, dim=0)
            if reduction == "mean":
                return (per_species_loss / per_species_weight).mean()
            elif reduction == "sum":
                return per_species_loss, counts, per_species_weight
            else:
                raise NotImplementedError("cannot handle this yet")
        else:
            if reduction == "mean":
                return scatter(
                    per_atom_loss, inverse_species_index, reduce="mean", dim=0
                ).mean()
            elif reduction == "sum":
                per_species_loss = scatter(
                    per_atom_loss, inverse_species_index, reduce="sum", dim=0
                )
                return per_atom_loss, counts, None
            else:
                raise NotImplementedError("cannot handle this yet")

    def accumulate_sum(self, new_sum, N, weight=None) -> torch.Tensor:
        """Accumulate a tally of samples into running statistics.

        Args:
            new_sum : the sum to tally up
            N: the corresponding number of samples
            weights

        Returns:
            the aggregated statistics _for this batch_. Accumulated statistics up to this point can be retreived with ``current_result()``.
        """
        self._state += (new_sum - N * self._state) / (self._n + N)
        if self.weight:
            self._weight += (weight - N * self._state) / (self._n + N)
        self._n += N

        # for the batch
        if self.weight:
            new_sum /= weight
        else:
            new_sum /= N

        if self._reduction == Reduction.RMS:
            new_sum = new_sum.sqrt()

        return new_sum

    def reset(self) -> None:
        """Forget all previously accumulated state."""
        self._n = 0
        if hasattr(self, "_state"):
            self._state.fill_(0.0)
            self._weight.fill_(0.0)
        else:
            self._state = torch.zeros(self._dim)
            self._weight = torch.zeros(self._dim)

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
