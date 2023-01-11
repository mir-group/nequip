from typing import List, Optional, Iterator
import math

import torch
from torch.utils.data import Sampler

from nequip.utils.torch_geometric import Batch, Data, Dataset


class Collater(object):
    """Collate a list of ``AtomicData``.

    Args:
        exclude_keys: keys to ignore in the input, not copying to the output
    """

    def __init__(
        self,
        exclude_keys: List[str] = [],
    ):
        self._exclude_keys = set(exclude_keys)

    @classmethod
    def for_dataset(
        cls,
        dataset,
        exclude_keys: List[str] = [],
    ):
        """Construct a collater appropriate to ``dataset``."""
        return cls(
            exclude_keys=exclude_keys,
        )

    def collate(self, batch: List[Data]) -> Batch:
        """Collate a list of data"""
        out = Batch.from_data_list(batch, exclude_keys=self._exclude_keys)
        return out

    def __call__(self, batch: List[Data]) -> Batch:
        """Collate a list of data"""
        return self.collate(batch)

    @property
    def exclude_keys(self):
        return list(self._exclude_keys)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        exclude_keys: List[str] = [],
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater.for_dataset(dataset, exclude_keys=exclude_keys),
            **kwargs,
        )


class PartialSampler(Sampler[int]):
    r"""Samples elements without replacement, but divided across a number of calls to `__iter__`.

    To ensure deterministic reproducibility and restartability, dataset permutations are generated
    from a combination of the overall seed and the epoch number. As a result, the caller must
    tell this sampler the epoch number before each time `__iter__` is called by calling
    `my_partial_sampler.step_epoch(epoch_number_about_to_run)` each time.

    Args:
        data_source (Dataset): dataset to sample from
        shuffle (bool): whether to shuffle the dataset each time the _entire_ dataset is consumed
        num_samples_per_segment (int): number of samples to draw in each call to `__iter__`.
            If `None`, defaults to `len(data_source)`. The entire dataset will be consumed in
            `ceil(len(data_source) / num_samples_per_segment)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Dataset
    num_samples_per_segment: int
    num_segments: int
    shuffle: bool
    _epoch: int
    _prev_epoch: int

    def __init__(
        self,
        data_source: Dataset,
        shuffle: bool = True,
        num_samples_per_segment: Optional[int] = None,
        generator=None,
    ) -> None:
        self.data_source = data_source
        self.shuffle = shuffle
        if num_samples_per_segment is None:
            num_samples_per_segment = len(data_source)
        self.num_samples_per_segment = num_samples_per_segment
        self.num_segments = int(
            math.ceil(self.num_samples_total / self.num_samples_per_segment)
        )
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

        full_epoch_i, segment_i = divmod(self._epoch, self.num_segments)
        if self.shuffle:
            temp_rng = torch.Generator()
            # Get new randomness for each _full_ time through the dataset
            # This is deterministic w.r.t. the combination of dataset seed and epoch number
            # Both of which persist across restarts
            # (initial_seed() is restored by set_state())
            temp_rng.manual_seed(self.generator.initial_seed() + full_epoch_i)
            full_order = torch.randperm(self.num_samples_total, generator=temp_rng)
        else:
            full_order = torch.arange(self.num_samples_total)

        this_segment_indexes = full_order[
            self.num_samples_per_segment
            * segment_i : self.num_samples_per_segment
            * (segment_i + 1)
        ]
        assert len(this_segment_indexes) > 0
        assert len(this_segment_indexes) <= self.num_samples_per_segment
        yield from this_segment_indexes

        self._prev_epoch = self._epoch

    def __len__(self) -> int:
        return self.num_samples_per_segment
