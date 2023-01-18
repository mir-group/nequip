from typing import Optional, List, Dict, Any, Tuple
import copy

import numpy as np

import ase
import ase.build
from ase.calculators.emt import EMT

from nequip.data import AtomicInMemoryDataset, AtomicData
from .transforms import TypeMapper


class EMTTestDataset(AtomicInMemoryDataset):
    """Test dataset with PBC based on the toy EMT potential included in ASE.

    Randomly generates (in a reproducable manner) a basic bulk with added
    Gaussian noise around equilibrium positions.

    In ASE units (eV/Å).
    """

    def __init__(
        self,
        root: str,
        supercell: Tuple[int, int, int] = (4, 4, 4),
        sigma: float = 0.1,
        element: str = "Cu",
        num_frames: int = 10,
        dataset_seed: int = 123456,
        file_name: Optional[str] = None,
        url: Optional[str] = None,
        AtomicData_options: Dict[str, Any] = {},
        include_frames: Optional[List[int]] = None,
        type_mapper: TypeMapper = None,
    ):
        # Set properties for hashing
        assert element in ("Cu", "Pd", "Au", "Pt", "Al", "Ni", "Ag")
        self.element = element
        self.sigma = sigma
        self.supercell = tuple(supercell)
        self.num_frames = num_frames
        self.dataset_seed = dataset_seed

        super().__init__(
            file_name=file_name,
            url=url,
            root=root,
            AtomicData_options=AtomicData_options,
            include_frames=include_frames,
            type_mapper=type_mapper,
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def raw_dir(self):
        return "raw"

    def get_data(self):
        rng = np.random.default_rng(self.dataset_seed)
        base_atoms = ase.build.bulk(self.element, "fcc").repeat(self.supercell)
        base_atoms.calc = EMT()
        orig_pos = copy.deepcopy(base_atoms.positions)
        datas = []
        for _ in range(self.num_frames):
            base_atoms.positions[:] = orig_pos
            base_atoms.positions += rng.normal(
                loc=0.0, scale=self.sigma, size=base_atoms.positions.shape
            )

            datas.append(
                AtomicData.from_ase(
                    base_atoms.copy(),
                    forces=base_atoms.get_forces(),
                    total_energy=base_atoms.get_potential_energy(),
                    stress=base_atoms.get_stress(voigt=False),
                    **self.AtomicData_options
                )
            )
        return datas
