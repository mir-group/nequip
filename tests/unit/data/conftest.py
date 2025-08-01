import numpy as np
import pytest

from ase.io import write

from nequip.data import AtomicDataDict
from nequip.data.dataset import ASEDataset, HDF5Dataset, EMTTestDataset, NPZDataset
from nequip.data.transforms import NeighborListTransform

from omegaconf import OmegaConf
from hydra.utils import instantiate
import tempfile

try:
    import h5py

    _H5PY_AVAILABLE = True
except ModuleNotFoundError:
    _H5PY_AVAILABLE = False

NATOMS = 10
MAX_ATOMIC_NUMBER: int = 5


@pytest.fixture(scope="module")
def npz():
    np.random.seed(0)
    natoms = NATOMS
    nframes = 8
    yield dict(
        positions=np.random.random((nframes, natoms, 3)),
        force=np.random.random((nframes, natoms, 3)),
        energy=np.random.random(nframes) * -600,
        Z=np.random.randint(1, MAX_ATOMIC_NUMBER, size=(nframes, natoms)),
    )


@pytest.fixture(scope="module")
def npz_for_NPZDataset():
    np.random.seed(0)
    natoms = NATOMS
    nframes = 8
    yield dict(
        R=np.random.random((nframes, natoms, 3)),
        F=np.random.random((nframes, natoms, 3)),
        E=np.random.random(nframes) * -600,
        z=np.random.randint(1, MAX_ATOMIC_NUMBER, size=natoms),
    )


@pytest.fixture(scope="module")
def npz_dataset(npz_for_NPZDataset):
    with tempfile.NamedTemporaryFile(suffix=".npz") as path:
        np.savez(path.name, **npz_for_NPZDataset)
        yield NPZDataset(
            file_path=path.name, transforms=[NeighborListTransform(r_max=3)]
        )


@pytest.fixture(scope="module")
def hdf5_dataset(npz):
    if not _H5PY_AVAILABLE:
        pytest.skip("h5py is not installed")
    with tempfile.NamedTemporaryFile(suffix=".hdf5") as path:
        f = h5py.File(path.name, "w")
        group = f.create_group("samples")
        group.create_dataset("atomic_numbers", data=npz["Z"], dtype=np.int8)
        group.create_dataset("pos", data=npz["positions"], dtype=np.float64)
        group.create_dataset("energy", data=npz["energy"], dtype=np.float64)
        group.create_dataset("forces", data=npz["force"], dtype=np.float64)
        yield HDF5Dataset(
            file_name=path.name,
            transforms=[NeighborListTransform(r_max=3)],
        )


@pytest.fixture(scope="module")
def ase_dataset(molecules):
    with tempfile.NamedTemporaryFile(suffix=".xyz") as fp:
        for atoms in molecules:
            write(fp.name, atoms, format="extxyz", append=True)
        yield ASEDataset(
            file_path=fp.name,
            transforms=[NeighborListTransform(r_max=3.0)],
            ase_args=dict(format="extxyz"),
        )


@pytest.fixture(scope="module")
def emt_dataset():
    yield EMTTestDataset(
        transforms=[NeighborListTransform(r_max=3.5)],
    )


@pytest.fixture(
    scope="module",
    params=["ase_dataset", "emt_dataset", "npz_dataset"]
    + (["hdf5_dataset"] if _H5PY_AVAILABLE else []),
)
def dataset(request):
    yield request.getfixturevalue(request.param)


@pytest.fixture(scope="function")
def TolueneDataModule(num_trainval_test, batch_size):
    with tempfile.TemporaryDirectory() as tmpdir:
        # this fixture is used in statistics tests
        config = """\
        data:
          _target_: nequip.data.datamodule.sGDML_CCSD_DataModule
          dataset: toluene
          data_source_dir: {}
          transforms:
            - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
              chemical_symbols: [C, H]
            - _target_: nequip.data.transforms.NeighborListTransform
              r_max: 4.0
          train_val_split: [0.8, 0.2]
          trainval_test_subset: [{}, {}]
          seed: 1234
          train_dataloader:
            _target_: torch.utils.data.DataLoader
            batch_size: {}
            shuffle: true
        """.format(
            tmpdir,
            num_trainval_test[0],
            num_trainval_test[1],
            # request.param,
            batch_size,
        )
        datacfg = OmegaConf.to_container(OmegaConf.create(config).data, resolve=True)
        data_module = instantiate(datacfg, _recursive_=False)
        data_module.prepare_data()
        data_module.setup("fit")
        dloader = data_module.train_dataloader()
        batch = AtomicDataDict.batched_from_list([data for data in dloader])
        yield data_module, dloader, batch
