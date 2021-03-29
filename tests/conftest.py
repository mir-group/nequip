import numpy as np
import pathlib
import pytest
import tempfile
import torch

from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from torch_geometric.data import Batch

from nequip.utils.test import set_irreps_debug
from nequip.data import AtomicData, ASEDataset

# For good practice, we *should* do this:
# See https://docs.pytest.org/en/stable/fixture.html#using-fixtures-from-other-projects
# pytest_plugins = ['e3nn.util.test']
# But doing so exposes float_tolerance to doctests, which don't support parametrized, autouse fixtures.
# Importing directly somehow only brings in the fixture later, preventing the issue.
from e3nn.util.test import float_tolerance

# Suppress linter errors
float_tolerance = float_tolerance


BENCHMARK_ROOT = pathlib.Path(__file__).parent / "../benchmark_data/"


@pytest.fixture(scope="session")
def temp_data(float_tolerance):
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture(scope="session")
def CH3CHO():
    atoms = molecule("CH3CHO")
    data = AtomicData.from_ase(atoms, r_max=2.0)
    return atoms, data


@pytest.fixture(scope="session")
def molecules():
    atoms_list = []
    for i in range(8):
        atoms = molecule("CH3CHO" if i % 2 == 0 else "H2")
        atoms.rattle()
        atoms.calc = SinglePointCalculator(
            energy=np.random.random(),
            forces=np.random.random((len(atoms), 3)),
            stress=None,
            magmoms=None,
            atoms=atoms,
        )
        atoms_list.append(atoms)
    return atoms_list


@pytest.fixture(scope="session")
def nequip_dataset(molecules, temp_data, float_tolerance):
    with tempfile.NamedTemporaryFile(suffix=".xyz") as fp:
        for atoms in molecules:
            write(fp.name, atoms, format="extxyz", append=True)
        a = ASEDataset(
            file_name=fp.name,
            root=temp_data,
            extra_fixed_fields={"r_max": 3.0},
            ase_args=dict(format="extxyz"),
        )
        yield a


@pytest.fixture(scope="session")
def atomic_batch(nequip_dataset):
    return Batch.from_data_list([nequip_dataset.data[0], nequip_dataset.data[1]])


# Use debug mode
set_irreps_debug(True)
