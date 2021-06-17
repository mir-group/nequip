import numpy as np

from os.path import dirname, basename, abspath

from ase import units
from ase.io import read

from nequip.data import AtomicDataDict, AtomicInMemoryDataset


class AspirinDataset(AtomicInMemoryDataset):
    """Aspirin DFT/CCSD(T) data """

    URL = "http://quantum-machine.org/gdml/data/npz/aspirin_ccsd.zip"
    FILE_NAME = "benchmark_data/aspirin_ccsd-train.npz"

    @property
    def raw_file_names(self):
        return [basename(self.file_name)]

    @property
    def raw_dir(self):
        return dirname(abspath(self.file_name))

    def get_data(self):
        data = np.load(self.raw_dir + "/" + self.raw_file_names[0])
        arrays = {
            AtomicDataDict.POSITIONS_KEY: data["R"],
            AtomicDataDict.FORCE_KEY: data["F"],
            AtomicDataDict.TOTAL_ENERGY_KEY: data["E"].reshape([-1, 1]),
        }
        fixed_fields = {
            AtomicDataDict.ATOMIC_NUMBERS_KEY: np.asarray(data["z"], dtype=np.int),
            AtomicDataDict.PBC_KEY: np.array([False, False, False]),
            #    AtomicDataDict.CELL_KEY: None,
        }
        return arrays, fixed_fields
