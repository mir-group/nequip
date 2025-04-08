# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Union, Callable, List, Tuple
import copy

import numpy as np
import torch

import ase
import ase.build
from ase.calculators.emt import EMT

from .. import AtomicDataDict
from ..dict import from_dict
from ._base_datasets import AtomicDataset


class EMTTestDataset(AtomicDataset):
    """Test dataset with PBC, based on the toy EMT potential included in ASE.

    Randomly generates (in a reproducable manner) a basic bulk with added
    Gaussian noise around equilibrium positions.

    In ASE units (eV, Å, eV/Å).

    Args:
        transforms (List[Callable]): list of data transforms
        supercell (Tuple[int, int, int]): supercell in each lattice vector direction
        sigma (float): standard deviation of Gaussian noise
        element (str): element supported by ASE's `EMT <https://wiki.fysik.dtu.dk/ase/ase/calculators/emt.html>`_ calculator
                       (supported elements: ``Cu``, ``Pd``, ``Au``, ``Pt``, ``Al``, ``Ni``, ``Ag``)
        num_frames (int): number of structures to be generated in the dataset
        seed (int): seed for the random Gaussian noise
    """

    def __init__(
        self,
        transforms: List[Callable] = [],
        supercell: Tuple[int, int, int] = (4, 4, 4),
        sigma: float = 0.1,
        element: str = "Cu",
        num_frames: int = 10,
        seed: int = 123456,
    ):
        super().__init__(transforms=transforms)
        assert element in ("Cu", "Pd", "Au", "Pt", "Al", "Ni", "Ag")
        self.element = element
        self.sigma = sigma
        self.supercell = tuple(supercell)
        self.num_frames = num_frames
        self.seed = seed

        # generate data
        base_atoms = ase.build.bulk(self.element, "fcc").repeat(self.supercell)
        base_atoms.calc = EMT()
        orig_pos = copy.deepcopy(base_atoms.positions)
        rng = np.random.default_rng(self.seed)
        self.data_list = []
        for idx in range(len(self)):
            base_atoms.positions[:] = orig_pos
            base_atoms.positions += rng.normal(
                loc=0.0, scale=self.sigma, size=base_atoms.positions.shape
            )
            self.data_list.append(
                from_dict(
                    {
                        "pos": base_atoms.positions,
                        "cell": np.array(base_atoms.get_cell()),
                        "pbc": base_atoms.get_pbc(),
                        "atomic_numbers": base_atoms.get_atomic_numbers(),
                        "forces": base_atoms.get_forces(),
                        "total_energy": base_atoms.get_potential_energy(),
                        "stress": base_atoms.get_stress(voigt=True),
                    }
                )
            )

    def __len__(self) -> int:
        return self.num_frames

    def get_data_list(
        self,
        indices: Union[List[int], torch.Tensor, slice],
    ) -> List[AtomicDataDict.Type]:
        if isinstance(indices, slice):
            return self.data_list[indices]
        else:
            return [self.data_list[index] for index in indices]
