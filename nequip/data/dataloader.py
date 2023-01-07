from typing import List

import torch

from nequip.utils.torch_geometric import Batch, Data


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
