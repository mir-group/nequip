# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import List, Tuple
import numpy as np
import pathlib
import pytest
import tempfile
import os

from ase.atoms import Atoms
from ase.build import molecule, bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

import torch

from nequip.data import AtomicDataDict, from_ase, compute_neighborlist_
from nequip.data.dataset import ASEDataset
from nequip.data.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
)
from nequip.utils.test import set_irreps_debug
from nequip.utils.global_state import set_global_state
from nequip.utils import dtype_to_name


# Sometimes we run parallel using pytest-xdist, and want to be able to use
# as many GPUs as are available
# https://pytest-xdist.readthedocs.io/en/latest/how-to.html#identifying-the-worker-process-during-a-test
_is_pytest_xdist: bool = os.environ.get("PYTEST_XDIST_WORKER", "master") != "master"
if _is_pytest_xdist and torch.cuda.is_available():
    _xdist_worker_rank: int = int(os.environ["PYTEST_XDIST_WORKER"].lstrip("gw"))
    _cuda_vis_devs = os.environ.get(
        "CUDA_VISIBLE_DEVICES",
        ",".join(str(e) for e in range(torch.cuda.device_count())),
    ).split(",")
    _cuda_vis_devs = [int(e) for e in _cuda_vis_devs]
    # set this for tests that run in this process
    _local_gpu_rank = _xdist_worker_rank % torch.cuda.device_count()
    torch.cuda.set_device(_local_gpu_rank)
    # set this for launched child processes
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_cuda_vis_devs[_local_gpu_rank])
    del _xdist_worker_rank, _cuda_vis_devs, _local_gpu_rank


@pytest.fixture(scope="session", params=["float32", "float64"])
def model_dtype(request):
    default_dtype = dtype_to_name(torch.get_default_dtype())
    if default_dtype != "float64":
        pytest.skip(
            f"found default_dtype={default_dtype} - only default_dtype=float64 will be tested"
        )
    return request.param


@pytest.fixture(scope="session", autouse=True)
def default_dtype():
    old_dtype = torch.get_default_dtype()
    # global dtype is always set to float64
    set_global_state()
    yield torch.get_default_dtype()
    torch.set_default_dtype(old_dtype)


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
def temp_data(default_dtype):
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture(scope="function")
def CH3CHO(CH3CHO_no_typemap) -> Tuple[Atoms, AtomicDataDict.Type]:
    atoms, data = CH3CHO_no_typemap
    tm = ChemicalSpeciesToAtomTypeMapper(
        chemical_symbols=["C", "O", "H"],
    )
    data = tm(data)
    return atoms, data


@pytest.fixture(scope="function")
def CH3CHO_no_typemap(default_dtype) -> Tuple[Atoms, AtomicDataDict.Type]:
    atoms = molecule("CH3CHO")
    data = compute_neighborlist_(
        from_ase(atoms),
        r_max=2.0,
    )
    return atoms, data


@pytest.fixture(scope="session")
def Cu_bulk(default_dtype) -> Tuple[Atoms, AtomicDataDict.Type]:
    atoms = bulk("Cu") * (2, 2, 1)
    atoms.rattle()
    data = compute_neighborlist_(from_ase(atoms), r_max=3.5, NL="ase")
    tm = ChemicalSpeciesToAtomTypeMapper(
        chemical_symbols=["Cu"],
    )
    data = tm(data)
    return atoms, data


@pytest.fixture(scope="session")
def molecules(default_dtype) -> List[Atoms]:
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
def nequip_dataset(molecules):
    with tempfile.NamedTemporaryFile(suffix=".xyz") as fp:
        for atoms in molecules:
            write(fp.name, atoms, format="extxyz", append=True)
        yield ASEDataset(
            transforms=[
                ChemicalSpeciesToAtomTypeMapper(
                    chemical_symbols=["H", "C", "O"],
                ),
                NeighborListTransform(r_max=3.0),
            ],
            file_path=fp.name,
            ase_args=dict(format="extxyz"),
        )


@pytest.fixture(scope="session")
def atomic_batch(nequip_dataset):
    return AtomicDataDict.batched_from_list([nequip_dataset[0], nequip_dataset[1]])


# Use debug mode
set_irreps_debug(True)
