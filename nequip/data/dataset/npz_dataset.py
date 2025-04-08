# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import numpy as np
import torch

from .. import AtomicDataDict
from ..dict import from_dict
from ._base_datasets import AtomicDataset

from typing import Union, Dict, List, Callable


class NPZDataset(AtomicDataset):
    """``AtomicDataset`` that loads data from an NPZ file following `sGDML <https://www.sgdml.org/#datasets>`_ conventions.
    It is also compatible with other datasets such as rMD-17, with a change in ``key_mapping``
    (the default ``key_mapping`` is set to be compatible with sGDML datasets).

    The ``NPZDataset`` avoids loading the whole dataset into memory.

    Args:
        file_path (str): path to npz file
        transforms (List[Callable]): list of data transforms
        key_mapping (Dict[str, str]): mapping of array names in the npz file to ``AtomicDataDict`` keys
    """

    def __init__(
        self,
        file_path: str,
        transforms: List[Callable] = [],
        key_mapping: Dict[str, str] = {
            "R": AtomicDataDict.POSITIONS_KEY,
            "z": AtomicDataDict.ATOMIC_NUMBERS_KEY,
            "E": AtomicDataDict.TOTAL_ENERGY_KEY,
            "F": AtomicDataDict.FORCE_KEY,
        },
    ):
        super().__init__(transforms=transforms)
        self.file_path = file_path
        self.key_mapping = key_mapping

        # use energy array to get num_frames (small to load into memory)
        E_key = None
        for k, v in key_mapping.items():
            if v == AtomicDataDict.TOTAL_ENERGY_KEY:
                E_key = k
        assert (
            E_key is not None
        ), "No key corresponding to `total_energy` found in npz dataset"
        with np.load(self.file_path, mmap_mode="r") as npz_data:
            self.num_frames = npz_data[E_key].shape[0]

    def __len__(self) -> int:
        return self.num_frames

    def get_data_list(
        self,
        indices: Union[List[int], torch.Tensor, slice],
    ) -> List[AtomicDataDict.Type]:

        if isinstance(indices, slice):
            indices = list(range(*indices.indices(self.num_frames)))

        # memory-map the file
        with np.load(self.file_path, mmap_mode="r") as npz_data:
            data_list = []
            for idx in indices:
                data_dict = {}
                for k, v in self.key_mapping.items():
                    # special case for now, may generalize if needed in the future
                    if v == AtomicDataDict.ATOMIC_NUMBERS_KEY:
                        data_dict[v] = npz_data[k]
                    else:
                        data_dict[v] = npz_data[k][idx]
                data_list.append(from_dict(data_dict))
            return data_list
