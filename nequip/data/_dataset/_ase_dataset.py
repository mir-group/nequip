from typing import Union, Dict, List, Optional, Callable, Any

import torch
import ase
import ase.io

from .. import AtomicDataDict
from ._base_datasets import AtomicDataset


class ASEDataset(AtomicDataset):
    """

    Args:
        ase_args (dict): arguments for ase.io.read
        include_keys (list): in addition to forces and energy, the keys that needs to
             be parsed into dataset
             The data stored in ase.atoms.Atoms.array has the lowest priority,
             and it will be overrided by data in ase.atoms.Atoms.info
             and ase.atoms.Atoms.calc.results. Optional
        key_mapping (dict): rename some of the keys to the value str. Optional

    Example: Given an atomic data stored in "H2.extxyz" that looks like below:

    ```H2.extxyz
    2
    Properties=species:S:1:pos:R:3 energy=-10 user_label=2.0 pbc="F F F"
     H       0.00000000       0.00000000       0.00000000
     H       0.00000000       0.00000000       1.02000000
    ```

    The yaml input should be

    ```
    dataset: ase
    dataset_file_name: H2.extxyz
    ase_args:
      format: extxyz
    include_keys:
      - user_label
    key_mapping:
      user_label: label0
    chemical_symbols:
      - H
    ```

    for VASP parser, the yaml input should be
    ```
    dataset: ase
    dataset_file_name: OUTCAR
    ase_args:
      format: vasp-out
    key_mapping:
      free_energy: total_energy
    chemical_symbols:
      - H
    ```

    """

    file_name: str
    ase_args: Dict[str, Any]

    def __init__(
        self,
        file_name: str,
        transforms: List[Callable] = [],
        ase_args: dict = {},
        include_keys: Optional[list] = [],
        key_mapping: Optional[Dict[str, str]] = {},
    ):
        super().__init__(transforms=transforms)
        self.file_name = file_name
        # process ase_args
        self.ase_args = {}
        self.ase_args.update(ase_args)
        assert "index" not in self.ase_args
        assert "filename" not in self.ase_args
        self.ase_args.update({"filename": file_name})

        # read file and construct list of AtomicDataDicts
        self.data_list: List[AtomicDataDict.Type] = []
        for atoms in ase.io.iread(**self.ase_args, parallel=False):
            self.data_list.append(
                AtomicDataDict.from_ase(
                    atoms=atoms, key_mapping=key_mapping, include_keys=include_keys
                )
            )

    def __len__(self) -> int:
        return len(self.data_list)

    def get_data_list(
        self,
        indices: Union[List[int], torch.Tensor, slice],
    ) -> List[AtomicDataDict.Type]:
        if isinstance(indices, slice):
            return self.data_list[indices]
        else:
            return [self.data_list[index] for index in indices]
