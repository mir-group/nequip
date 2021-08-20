from typing import List

import torch

from torch_geometric.data import Batch, Data


class Collater(object):
    """Collate a list of ``AtomicData``.

    Args:
        fixed_fields: which fields are fixed fields
        exclude_keys: keys to ignore in the input, not copying to the output
    """

    def __init__(
        self,
        fixed_fields: List[str] = [],
        exclude_keys: List[str] = [],
    ):
        self.fixed_fields = fixed_fields
        self._exclude_keys = set(exclude_keys)

    @classmethod
    def for_dataset(
        cls,
        dataset,
        exclude_keys: List[str] = [],
    ):
        """Construct a collater appropriate to ``dataset``.

        All kwargs besides ``fixed_fields`` are passed through to the constructor.
        """
        return cls(
            fixed_fields=list(getattr(dataset, "fixed_fields", {}).keys()),
            exclude_keys=exclude_keys,
        )

    def collate(self, batch: List[Data]) -> Batch:
        """Collate a list of data"""
        # For fixed fields, we need to batch those that are per-node or
        # per-edge, since they need to be repeated in order to have the same
        # number of nodes/edges as the full batch graph.
        # For fixed fields that are per-example, however — those with __cat_dim__
        # of None — we can just put one copy over the whole batch graph.
        # Figure out which ones those are:
        new_dim_fixed = set()
        for f in self.fixed_fields:
            if batch[0].__cat_dim__(f, None) is None:
                new_dim_fixed.add(f)
        # TODO: cache ^ and the batched versions of fixed fields for various batch sizes if necessary for performance
        out = Batch.from_data_list(
            batch, exclude_keys=self._exclude_keys.union(new_dim_fixed)
        )
        for f in new_dim_fixed:
            if f in self._exclude_keys:
                continue
            out[f] = batch[0][f]
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
