# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from nequip.data import AtomicDataDict


class DatasetIndexTransform(torch.nn.Module):
    """Add a dataset index to ``AtomicDataDict``."""

    def __init__(self, dataset_index: int):
        super().__init__()
        self.dataset_index = dataset_index

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[AtomicDataDict.DATASET_KEY] = torch.tensor(
            self.dataset_index,
            dtype=torch.long,
        ).view(-1, 1)
        return data
