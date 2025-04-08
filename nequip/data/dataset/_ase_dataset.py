# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
import ase
import ase.io

from .. import AtomicDataDict
from ..ase import from_ase
from ._base_datasets import AtomicDataset

from typing import Union, Dict, List, Optional, Callable, Any

# TODO: link "standard keys" under `include_keys` to docs


class ASEDataset(AtomicDataset):
    r"""``AtomicDataset`` for `ASE <https://wiki.fysik.dtu.dk/ase/ase/io/io.html>`_-readable file formats.

    Args:
        file_path (str): path to ASE-readable file
        transforms (List[Callable]): list of data transforms
        ase_args (Dict[str, Any]): arguments for ``ase.io.iread`` (see `here <https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.iread>`_)
        include_keys (List[str]): the keys that needs to be parsed into dataset in addition to standard keys. The data stored in ``ase.atoms.Atoms.array`` has the lowest priority, and it will be overrided by data in ``ase.atoms.Atoms.info`` and ``ase.atoms.Atoms.calc.results``
        exclude_keys (List[str]): list of keys that may be present in the ASE-readable file but the user wishes to exclude
        key_mapping (Dict[str, str]): mapping of ``ase`` keys to ``AtomicDataDict`` keys
    """

    def __init__(
        self,
        file_path: str,
        transforms: List[Callable] = [],
        ase_args: Dict[str, Any] = {},
        include_keys: Optional[List[str]] = [],
        exclude_keys: Optional[List[str]] = [],
        key_mapping: Optional[Dict[str, str]] = {},
    ):
        super().__init__(transforms=transforms)
        self.file_path = file_path
        # process ase_args
        self.ase_args = {}
        self.ase_args.update(ase_args)
        assert "index" not in self.ase_args
        assert "filename" not in self.ase_args
        self.ase_args.update({"filename": self.file_path})

        # read file and construct list of AtomicDataDicts
        self.data_list: List[AtomicDataDict.Type] = []
        for atoms in ase.io.iread(**self.ase_args, parallel=False):
            self.data_list.append(
                from_ase(
                    atoms=atoms,
                    key_mapping=key_mapping,
                    include_keys=include_keys,
                    exclude_keys=exclude_keys,
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
