import numpy as np
import torch

from .. import AtomicDataDict
from ._base_datasets import AtomicDataset

from typing import Union, Dict, List, Callable


class sGDMLNPZDataset(AtomicDataset):
    """``AtomicDataset`` that loads data from an NPZ file following sGDML conventions.

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

        # TODO: need to check that key mapping is unique?
        pos_key = None
        for k, v in key_mapping.items():
            if v == AtomicDataDict.POSITIONS_KEY:
                pos_key = k
        assert pos_key is not None, "No key indicating position found in npz dataset"

        self.key_mapping = key_mapping
        npz_data = np.load(self.file_path)
        num_frames = npz_data[pos_key].shape[0]

        # set up list of AtomicDataDicts
        self.data_list = []
        for idx in range(num_frames):
            data_dict = {}
            for k, v in self.key_mapping.items():
                # special case for now, may generalize if needed in the future
                if v == AtomicDataDict.ATOMIC_NUMBERS_KEY:
                    data_dict[v] = npz_data[k]
                else:
                    data_dict[v] = npz_data[k][idx]
            self.data_list.append(AtomicDataDict.from_dict(data_dict))

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
