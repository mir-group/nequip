import numpy as np

from ase import units
from ase.io import read

from nequip.data import ASEDataset, AtomicDataDict


class ChengWaterDataset(ASEDataset):
    """Cheng Water Dataset

    TODO: energies
    """

    URL = None
    FILE_NAME = "benchmark_data/dataset_1593.xyz"
    FORCE_FIXED_KEYS = [AtomicDataDict.PBC_KEY]

    def download(self):
        pass
