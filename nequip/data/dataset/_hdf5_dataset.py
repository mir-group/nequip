# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Union, Dict, List, Callable

import torch

from .. import AtomicDataDict
from ..dict import from_dict
from ._base_datasets import AtomicDataset


class HDF5Dataset(AtomicDataset):
    """``AtomicDataset`` that loads data from a HDF5 file.

    This class is useful for very large datasets that cannot fit in memory. It
    efficiently loads data from disk as needed without everything needing to be
    in memory at once.

    To use this, ``file_name`` should point to the HDF5 file, or alternatively a
    semicolon separated list of multiple files.  Each group in the file contains
    samples that all have the same number of atoms.  Typically there is one
    group for each unique number of atoms, but that is not required.  Each group
    should contain arrays whose length equals the number of samples, one for each
    type of data.  The names of the arrays can be specified with ``key_mapping``.

    Args:
        file_name (str): a semicolon separated list of HDF5 files.
        transforms (List[Callable]): list of data transforms
        key_mapping (Dict[str, str]): mapping of array names in the HDF5 file to ``AtomicDataDict`` keys
    """

    def __init__(
        self,
        file_name: str,
        transforms: List[Callable] = [],
        key_mapping: Dict[str, str] = {
            "pos": AtomicDataDict.POSITIONS_KEY,
            "energy": AtomicDataDict.TOTAL_ENERGY_KEY,
            "forces": AtomicDataDict.FORCE_KEY,
            "atomic_numbers": AtomicDataDict.ATOMIC_NUMBERS_KEY,
            "types": AtomicDataDict.ATOM_TYPE_KEY,
        },
    ):
        super().__init__(transforms=transforms)
        self.file_name = file_name
        self.key_mapping = key_mapping
        self.index = None
        self.num_frames = 0

        import h5py

        files = [h5py.File(f, "r") for f in self.file_name.split(";")]
        for file in files:
            for group_name in file:
                for key in self.key_mapping.keys():
                    if key in file[group_name]:
                        self.num_frames += len(file[group_name][key])
                        break
            file.close()

    def setup_index(self):
        import h5py

        files = [h5py.File(f, "r") for f in self.file_name.split(";")]
        self.has_forces = False
        self.index = []
        for file in files:
            for group_name in file:
                group = file[group_name]
                values = [None] * len(self.key_mapping.keys())
                samples = 0
                for i, key in enumerate(self.key_mapping.keys()):
                    if key in group:
                        values[i] = group[key]
                        samples = len(values[i])
                for i in range(samples):
                    self.index.append(tuple(values + [i]))

    def __len__(self) -> int:
        return self.num_frames

    def get_data_list(
        self,
        indices: Union[List[int], torch.Tensor, slice],
    ) -> List[AtomicDataDict.Type]:
        if self.index is None:
            self.setup_index()
        if isinstance(indices, slice):
            indices = range(len(self))[indices]
        return [self._get_data(index) for index in indices]

    def _get_data(self, idx: int) -> AtomicDataDict:
        data = self.index[idx]
        i = data[-1]
        data_dict = {}
        for j, value in enumerate(self.key_mapping.values()):
            if data[j] is not None:
                data_dict[value] = data[j][i]
        return from_dict(data_dict)
