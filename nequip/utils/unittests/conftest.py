from typing import List, Tuple
import numpy as np
import pathlib
import pytest
import tempfile
import os

from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

import torch

from nequip.utils.test import set_irreps_debug
from nequip.data import AtomicData, ASEDataset
from nequip.data.transforms import TypeMapper
from nequip.utils.torch_geometric import Batch
from nequip.utils._global_options import _set_global_options
from nequip.utils.misc import dtype_from_name

if "NEQUIP_NUM_TASKS" not in os.environ:
    # Test parallelization, but don't waste time spawning tons of workers if lots of cores available
    os.environ["NEQUIP_NUM_TASKS"] = "2"

# The default float tolerance
FLOAT_TOLERANCE = {
    t: torch.as_tensor(v, dtype=dtype_from_name(t))
    for t, v in {"float32": 1e-3, "float64": 1e-10}.items()
}


@pytest.fixture(scope="session", autouse=True, params=["float32", "float64"])
def float_tolerance(request):
    """Run all tests with various PyTorch default dtypes.

    This is a session-wide, autouse fixture — you only need to request it explicitly if a test needs to know the tolerance for the current default dtype.

    Returns
    --------
        A precision threshold to use for closeness tests.
    """
    old_dtype = torch.get_default_dtype()
    dtype = request.param
    _set_global_options({"default_dtype": dtype})
    yield FLOAT_TOLERANCE[dtype]
    _set_global_options(
        {
            "default_dtype": {torch.float32: "float32", torch.float64: "float64"}[
                old_dtype
            ]
        }
    )


# - Ampere and TF32 -
# Many of the tests for NequIP involve numerically checking
# algebraic properties— normalization, equivariance,
# continuity, etc.
# With the added numerical noise of TF32, some of those tests fail
# with the current (and usually generous) thresholds.
#
# Thus we go on the assumption that PyTorch + NVIDIA got everything
# right, that this setting DOES NOT AFFECT the model outputs except
# for increased numerical noise, and only test without it.
#
# TODO: consider running tests with and without
# TODO: check how much thresholds have to be changed to accomidate TF32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@pytest.fixture(scope="session")
def BENCHMARK_ROOT():
    return pathlib.Path(__file__).parent / "../benchmark_data/"


@pytest.fixture(scope="session")
def temp_data(float_tolerance):
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture(scope="session")
def CH3CHO(CH3CHO_no_typemap) -> Tuple[Atoms, AtomicData]:
    atoms, data = CH3CHO_no_typemap
    tm = TypeMapper(chemical_symbol_to_type={"C": 0, "O": 1, "H": 2})
    data = tm(data)
    return atoms, data


@pytest.fixture(scope="session")
def CH3CHO_no_typemap(float_tolerance) -> Tuple[Atoms, AtomicData]:
    atoms = molecule("CH3CHO")
    data = AtomicData.from_ase(atoms, r_max=2.0)
    return atoms, data


@pytest.fixture(scope="session")
def molecules() -> List[Atoms]:
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
            type_mapper=TypeMapper(chemical_symbol_to_type={"H": 0, "C": 1, "O": 2}),
        )
        yield a


@pytest.fixture(scope="session")
def atomic_batch(nequip_dataset):
    return Batch.from_data_list([nequip_dataset[0], nequip_dataset[1]])


@pytest.fixture(scope="function")
def per_species_set():
    dtype = torch.get_default_dtype()
    rng = torch.Generator().manual_seed(127)
    mean_min = 1
    mean_max = 100
    std = 20
    n_sample = 1000
    n_species = 9
    ref_mean = torch.rand((n_species), generator=rng) * (mean_max - mean_min) + mean_min
    t_mean = torch.ones((n_sample, 1)) * ref_mean.reshape([1, -1])
    ref_std = torch.rand((n_species), generator=rng) * std
    t_std = torch.ones((n_sample, 1)) * ref_std.reshape([1, -1])
    E = torch.normal(t_mean, t_std, generator=rng)
    return ref_mean.to(dtype), ref_std.to(dtype), E.to(dtype), n_sample, n_species


# Use debug mode
set_irreps_debug(True)
