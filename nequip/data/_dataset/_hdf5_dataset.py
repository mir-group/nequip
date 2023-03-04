from typing import Dict, Any, List, Callable, Union, Optional
import numpy as np

import torch

from .. import (
    AtomicData,
    AtomicDataDict,
)
from ..transforms import TypeMapper
from ._base_datasets import AtomicDataset


class HDF5Dataset(AtomicDataset):
    """A dataset that loads data from a HDF5 file.

    This class is useful for very large datasets that cannot fit in memory.  It
    efficiently loads data from disk as needed without everything needing to be
    in memory at once.

    To use this, ``file_name`` should point to the HDF5 file, or alternatively a
    semicolon separated list of multiple files.  Each group in the file contains
    samples that all have the same number of atoms.  Typically there is one
    group for each unique number of atoms, but that is not required.  Each group
    should contain arrays whose length equals the number of samples, one for each
    type of data.  The names of the arrays can be specified with ``key_mapping``.

    Args:
        key_mapping (Dict[str, str]): mapping of array names in the HDF5 file to ``AtomicData`` keys
        file_name (string): a semicolon separated list of HDF5 files.
    """
    def __init__(
        self,
        root: str,
        key_mapping: Dict[str, str] = {
            "pos": AtomicDataDict.POSITIONS_KEY,
            "energy": AtomicDataDict.TOTAL_ENERGY_KEY,
            "forces": AtomicDataDict.FORCE_KEY,
            "atomic_numbers": AtomicDataDict.ATOMIC_NUMBERS_KEY,
            "types": AtomicDataDict.ATOM_TYPE_KEY,
        },
        file_name: Optional[str] = None,
        AtomicData_options: Dict[str, Any] = {},
        type_mapper: Optional[TypeMapper] = None,
    ):
        super().__init__(root=root, type_mapper=type_mapper)
        self.key_mapping = key_mapping
        self.key_list = list(key_mapping.keys())
        self.file_name = file_name
        self.r_max = AtomicData_options["r_max"]
        self.index = None
        self.num_frames = 0
        import h5py

        files = [h5py.File(f, "r") for f in self.file_name.split(";")]
        for file in files:
            for group_name in file:
                for key in self.key_list:
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
                values = [None]*len(self.key_list)
                samples = 0
                for i, key in enumerate(self.key_list):
                    if key in group:
                        values[i] = np.array(group[key])
                        samples = len(values[i])
                for i in range(samples):
                    self.index.append(tuple(values+[i]))

    def len(self) -> int:
        return self.num_frames

    def get(self, idx: int) -> AtomicData:
        if self.index is None:
            self.setup_index()
        data = self.index[idx]
        i = data[-1]
        args = {"r_max": self.r_max}
        for j, key in enumerate(self.key_list):
            if data[j][i] is not None:
                args[self.key_mapping[self.key_list[j]]] = data[j][i]
        return AtomicData.from_points(**args)

    def statistics(
        self,
        fields: List[Union[str, Callable]],
        modes: List[str],
        stride: int = 1,
        unbiased: bool = True,
        kwargs: Optional[Dict[str, dict]] = {},
    ) -> List[tuple]:
        assert len(modes) == len(fields)
        # TODO: use RunningStats
        if len(fields) == 0:
            return []
        if self.index is None:
            self.setup_index()
        results = []
        for field, mode in zip(fields, modes):
            count = 0
            if mode == "rms":
                total = 0.0
            elif mode in ("mean_std", "per_atom_mean_std"):
                total = [0.0, 0.0]
            else:
                raise NotImplementedError(f"Analysis mode '{mode}' is not implemented")
            for index in range(0, self.len(), stride):
                data = self.index[index]
                i = data[-1]
                if field == AtomicDataDict.FORCE_KEY:
                    values = data[3][i]
                elif field == AtomicDataDict.TOTAL_ENERGY_KEY:
                    values = data[2][i]
                elif callable(field):
                    values, _ = field(self.get(index))
                    values = np.asarray(values)
                else:
                    raise RuntimeError(
                        f"The field key `{field}` is not present in this dataset"
                    )
                if mode == "rms":
                    total += np.sum(values * values)
                    count += len(values.flatten())
                else:
                    length = len(values.flatten())
                    if mode == "per_atom_mean_std":
                        values /= len(data[0][i])
                    sample_mean = np.mean(values)
                    new_mean = (total[0] * count + sample_mean * length) / (
                        count + length
                    )
                    total[1] += (
                        length * (sample_mean - total[0]) * (sample_mean - new_mean)
                    )
                    total[0] = new_mean
                    count += length
            if mode == "rms":
                results.append(torch.tensor((np.sqrt(total / count),)))
            else:
                results.append(
                    (torch.tensor(total[0]), torch.tensor(np.sqrt(total[1] / count)))
                )
        return results
