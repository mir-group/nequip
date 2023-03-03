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
    def __init__(
        self,
        root: str,
        file_name: Optional[str] = None,
        extra_fixed_fields: Dict[str, Any] = {},
        type_mapper: Optional[TypeMapper] = None,
    ):
        super().__init__(root=root, type_mapper=type_mapper)
        self.file_name = file_name
        self.r_max = extra_fixed_fields["r_max"]
        self.index = None
        self.num_molecules = 0
        import h5py

        files = [h5py.File(f, "r") for f in self.file_name.split(";")]
        for file in files:
            for group_name in file:
                self.num_molecules += len(file[group_name]["energy"])
            file.close()

    def setup_index(self):
        import h5py

        files = [h5py.File(f, "r") for f in self.file_name.split(";")]
        self.has_forces = False
        self.index = []
        for file in files:
            for group_name in file:
                group = file[group_name]
                types = np.array(group["types"])
                pos = np.array(group["pos"])
                energy = np.array(group["energy"])
                if "forces" in group:
                    self.has_forces = True
                    forces = -np.array(group["forces"])
                    for i in range(len(energy)):
                        self.index.append((types, pos, energy, forces, i))
                else:
                    for i in range(len(energy)):
                        self.index.append((types, pos, energy, i))

    def len(self) -> int:
        return self.num_molecules

    def get(self, idx: int) -> AtomicData:
        if self.index is None:
            self.setup_index()
        data = self.index[idx]
        i = data[-1]
        args = {
            "pos": data[1][i],
            "r_max": self.r_max,
            AtomicDataDict.ATOM_TYPE_KEY: data[0][i],
            AtomicDataDict.TOTAL_ENERGY_KEY: data[2][i],
        }
        if self.has_forces:
            args[AtomicDataDict.FORCE_KEY] = data[3][i]
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
