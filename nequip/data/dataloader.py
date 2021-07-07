from typing import List

import torch

from torch_geometric.data import Batch, Data


class Collater(object):
    def __init__(self, fixed_fields=[], exclude_keys=[]):
        self.fixed_fields = fixed_fields
        self.exclude_keys = exclude_keys
        self._exclude_keys = set(exclude_keys)

    @classmethod
    def for_dataset(cls, dataset, exclude_keys=[]):
        return cls(
            fixed_fields=list(getattr(dataset, "fixed_fields", {}).keys()),
            exclude_keys=exclude_keys,
        )

    def collate(self, batch: List[Data]):
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

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, exclude_keys=[], **kwargs):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater.for_dataset(dataset, exclude_keys=exclude_keys),
            **kwargs
        )
