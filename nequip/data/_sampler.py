# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Optional, Iterator

import torch
from torch.utils.data import Sampler


class PartialSampler(Sampler[int]):
    r"""Samples elements without replacement, but divided across a number of calls to `__iter__`.

    To ensure deterministic reproducibility and restartability, dataset permutations are generated
    from a combination of the overall seed and the epoch number. As a result, the caller must
    tell this sampler the epoch number before each time `__iter__` is called by calling
    `my_partial_sampler.step_epoch(epoch_number_about_to_run)` each time.

    This sampler decouples epochs from the dataset size and cycles through the dataset over as
    many (partial) epochs as it may take. As a result, the _dataset_ epoch can change partway
    through a training epoch.

    Args:
        data_source (Dataset): dataset to sample from
        shuffle (bool): whether to shuffle the dataset each time the _entire_ dataset is consumed
        num_samples_per_epoch (int): number of samples to draw in each call to `__iter__`.
            If `None`, defaults to `len(data_source)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: torch.utils.data.Dataset
    num_samples_per_epoch: int
    shuffle: bool
    _epoch: int
    _prev_epoch: int

    def __init__(
        self,
        data_source: torch.utils.data.Dataset,
        shuffle: bool = True,
        num_samples_per_epoch: Optional[int] = None,
        generator=None,
    ) -> None:
        self.data_source = data_source
        self.shuffle = shuffle
        if num_samples_per_epoch is None:
            num_samples_per_epoch = self.num_samples_total
        self.num_samples_per_epoch = num_samples_per_epoch
        assert self.num_samples_per_epoch <= self.num_samples_total
        assert self.num_samples_per_epoch >= 1
        self.generator = generator
        self._epoch = None
        self._prev_epoch = None

    @property
    def num_samples_total(self) -> int:
        # dataset size might change at runtime
        return len(self.data_source)

    def step_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        assert self._epoch is not None
        assert (self._prev_epoch is None) or (self._epoch == self._prev_epoch + 1)
        assert self._epoch >= 0

        full_epoch_i, start_sample_i = divmod(
            # how much data we've already consumed:
            self._epoch * self.num_samples_per_epoch,
            # how much data there is the dataset:
            self.num_samples_total,
        )

        if self.shuffle:
            temp_rng = torch.Generator()
            # Get new randomness for each _full_ time through the dataset
            # This is deterministic w.r.t. the combination of dataset seed and epoch number
            # Both of which persist across restarts
            # (initial_seed() is restored by set_state())
            temp_rng.manual_seed(self.generator.initial_seed() + full_epoch_i)
            full_order_this = torch.randperm(self.num_samples_total, generator=temp_rng)
            # reseed the generator for the _next_ epoch to get the shuffled order of the
            # _next_ dataset epoch to pad out this one for completing any partial batches
            # at the end:
            temp_rng.manual_seed(self.generator.initial_seed() + full_epoch_i + 1)
            full_order_next = torch.randperm(self.num_samples_total, generator=temp_rng)
            del temp_rng
        else:
            full_order_this = torch.arange(self.num_samples_total)
            # without shuffling, the next epoch has the same sampling order as this one:
            full_order_next = full_order_this

        full_order = torch.cat((full_order_this, full_order_next), dim=0)
        del full_order_next, full_order_this

        this_segment_indexes = full_order[
            start_sample_i : start_sample_i + self.num_samples_per_epoch
        ]
        # because we cycle into indexes from the next dataset epoch,
        # we should _always_ be able to get num_samples_per_epoch
        assert len(this_segment_indexes) == self.num_samples_per_epoch
        yield from this_segment_indexes

        self._prev_epoch = self._epoch

    def __len__(self) -> int:
        return self.num_samples_per_epoch
