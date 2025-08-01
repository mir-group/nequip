import pytest

import numpy as np
from pathlib import Path

import ase
import ase.io
import ase.data

from nequip.data.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
)
from nequip.ase import NequIPCalculator
from nequip.utils.unittests.model_tests import BaseEnergyModelTests


_ZBL_TEST_RMAX: float = 8.0  # see zbl_data.lmps


class TestZBLModel(BaseEnergyModelTests):
    @pytest.fixture
    def strict_locality(self):
        return True

    @pytest.mark.skip(reason="Skip compile tests for ZBL models.")
    def test_nequip_compile(self):
        pass

    @pytest.mark.skip(reason="Skip LAMMPS ML-IAP tests for ZBL models.")
    def test_mliap_integration(self):
        pass

    @pytest.fixture(scope="class")
    def config(self):
        config = {
            "_target_": "nequip.model.ZBLPairPotential",
            "seed": 123,
            "r_max": _ZBL_TEST_RMAX + 1,  # To make cutoff envelope irrelevant
            "polynomial_cutoff_p": 80,  # almost a step function
            "type_names": ["H", "O", "C", "N", "Cu", "Au"],
            "chemical_species": ["H", "O", "C", "N", "Cu", "Au"],
            "units": "metal",
        }
        return config

    def test_lammps_repro(self, model, device):
        ZBL_model, config, _ = model

        # run test only if model is float64
        if config["model_dtype"] == "float64":
            transforms = [
                ChemicalSpeciesToAtomTypeMapper(
                    chemical_symbols=config["chemical_species"],
                ),
                NeighborListTransform(r_max=_ZBL_TEST_RMAX),
            ]
            ZBL_model.eval()
            # make test system of two atoms:
            atoms = ase.Atoms(positions=np.zeros((2, 3)), symbols=["H", "H"])
            atoms.calc = NequIPCalculator(
                ZBL_model, device=device, transforms=transforms
            )
            # == load precomputed reference data ==
            # To regenerate this data, run
            # $ lmp -in zbl_data.lmps
            # $ python -c "import numpy as np; d = np.loadtxt('zbl.dat', skiprows=1); np.save('zbl.npy', d)"
            refdata = np.load(Path(__file__).parent / "zbl.npy")
            for r, Zi, Zj, pe, fxi, fxj in refdata:
                if r >= _ZBL_TEST_RMAX:
                    continue
                atoms.positions[1, 0] = r
                atoms.set_atomic_numbers([int(Zi), int(Zj)])
                # ZBL blows up for atoms being close, so the numerics differ to ours
                # 1e-5 == 0.01 meV / Å
                assert np.allclose(atoms.get_forces()[0, 0], fxi, atol=1e-5)
                assert np.allclose(atoms.get_forces()[1, 0], fxj, atol=1e-5)
                # 1e-4 == 0.1 meV system, 0.05 meV / atom
                assert np.allclose(atoms.get_potential_energy(), pe, atol=1e-4)
