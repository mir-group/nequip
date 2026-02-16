import pytest

import numpy as np
from pathlib import Path

import ase

from nequip.data.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
)
from nequip.ase import NequIPCalculator
from nequip.utils.unittests.model_tests_basic import EnergyModelTestsMixin


_ZBL_TEST_RMAX: float = 8.0  # see zbl_data.lmps


class TestZBLModel(EnergyModelTestsMixin):
    """ZBL pair potential tests.

    Inherits from EnergyModelTestsMixin directly to get basic + energy tests.
    """

    @pytest.fixture(scope="class")
    def equivariance_tol(self, model_dtype):
        # CI fails with at most 6e-8 errors for fp64 models for PyTorch 2.9.1
        # so fp64 tol bumped to 1e-7
        # With PyTorch 2.10.0:
        # ```
        # FAILED tests/unit/model/test_pair/test_zbl.py::TestZBLModel::test_equivariance[float64-bulk-cpu] - AssertionError: Equivariance test of GraphModel failed:
        # (parity_k=1, did_translate=True , field=virial                )     -> max error=1.707e-07  FAIL
        # ```
        # so we bump fp64 tol to 5e-7
        # NOTE: this test is looking pretty flaky for the virials, might be useful to look into it
        # to try and understand why the numerics are so flaky (may be due to ops in the ZBL functional form)
        return {"float32": 1e-3, "float64": 5e-7}[model_dtype]

    @pytest.fixture
    def strict_locality(self):
        return True

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
                    model_type_names=config["type_names"],
                    chemical_species_to_atom_type_map={
                        s: s for s in config["chemical_species"]
                    },
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
                np.testing.assert_allclose(atoms.get_forces()[0, 0], fxi, atol=1e-5)
                np.testing.assert_allclose(atoms.get_forces()[1, 0], fxj, atol=1e-5)
                # 1e-4 == 0.1 meV system, 0.05 meV / atom
                np.testing.assert_allclose(atoms.get_potential_energy(), pe, atol=1e-4)
