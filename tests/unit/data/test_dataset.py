import numpy as np
import pytest
import tempfile
import torch

from os.path import isdir, isfile

from ase.io import write

from nequip.data import (
    AtomicDataDict,
    AtomicInMemoryDataset,
    NpzDataset,
    ASEDataset,
)
from nequip.utils import dataset_from_config, Config


@pytest.fixture(scope="module")
def ase_file(molecules):
    with tempfile.NamedTemporaryFile(suffix=".xyz") as fp:
        for atoms in molecules:
            write(fp.name, atoms, format="extxyz", append=True)
        yield fp.name


@pytest.fixture(scope="session")
def npz():
    natoms = 3
    nframes = 4
    yield dict(
        positions=np.random.random((nframes, natoms, 3)),
        force=np.random.random((nframes, natoms, 3)),
        energy=np.random.random(nframes),
        Z=np.random.randint(1, 8, size=(nframes, natoms)),
    )


@pytest.fixture(scope="session")
def npz_data(npz):
    with tempfile.NamedTemporaryFile(suffix=".npz") as path:
        np.savez(path.name, **npz)
        yield path.name


@pytest.fixture(scope="session")
def npz_dataset(npz_data, temp_data):
    a = NpzDataset(
        file_name=npz_data,
        root=temp_data + "/test_dataset",
        extra_fixed_fields={"r_max": 3},
    )
    yield a


@pytest.fixture(scope="function")
def root():
    with tempfile.TemporaryDirectory(prefix="datasetroot") as path:
        yield path


class TestInit:
    def test_init(self):
        with pytest.raises(NotImplementedError) as excinfo:
            a = AtomicInMemoryDataset(root=None)
        assert str(excinfo.value) == ""

    def test_npz(self, npz_data, root):
        g = NpzDataset(file_name=npz_data, root=root, extra_fixed_fields={"r_max": 3.0})
        assert isdir(g.root)
        assert isdir(g.processed_dir)
        assert isfile(g.processed_dir + "/data.pth")

    def test_ase(self, ase_file, root):
        a = ASEDataset(
            file_name=ase_file,
            root=root,
            extra_fixed_fields={"r_max": 3.0},
            ase_args=dict(format="extxyz"),
        )
        assert isdir(a.root)
        assert isdir(a.processed_dir)
        assert isfile(a.processed_dir + "/data.pth")


class TestStatistics:
    @pytest.mark.xfail(
        reason="Current subset hack doesn't support statistics of non-per-node callable"
    )
    def test_callable(self, npz_dataset, npz):
        # Get componentwise statistics
        ((f_mean, f_std),) = npz_dataset.statistics(
            [lambda d: torch.flatten(d[AtomicDataDict.FORCE_KEY])]
        )
        n_ex, n_at, _ = npz["force"].shape
        f_raveled = npz["force"].reshape((n_ex * n_at * 3,))
        assert np.allclose(np.mean(f_raveled), f_mean)
        # By default we follow torch convention of defaulting to the unbiased std
        assert np.allclose(np.std(f_raveled, ddof=1), f_std)

    def test_statistics(self, npz_dataset, npz):

        (eng_mean, eng_std), (Z_unique, Z_count) = npz_dataset.statistics(
            [AtomicDataDict.TOTAL_ENERGY_KEY, AtomicDataDict.ATOMIC_NUMBERS_KEY]
        )

        eng = npz["energy"]
        assert np.allclose(eng_mean, np.mean(eng))
        # By default we follow torch convention of defaulting to the unbiased std
        assert np.allclose(eng_std, np.std(eng, ddof=1))

        if isinstance(Z_count, torch.Tensor):
            Z_count = Z_count.numpy()
            Z_unique = Z_unique.numpy()

        uniq, count = np.unique(npz["Z"].ravel(), return_counts=True)
        assert np.all(Z_unique == uniq)
        assert np.all(Z_count == count)

    def test_with_subset(self, npz_dataset, npz):

        dataset = npz_dataset.index_select([0])

        ((Z_unique, Z_count), (force_rms,)) = dataset.statistics(
            [AtomicDataDict.ATOMIC_NUMBERS_KEY, AtomicDataDict.FORCE_KEY],
            modes=["count", "rms"],
        )
        print("npz", npz["Z"])

        uniq, count = np.unique(npz["Z"][0].ravel(), return_counts=True)
        assert np.all(Z_unique.numpy() == uniq)
        assert np.all(Z_count.numpy() == count)

        assert np.allclose(
            force_rms.numpy(), np.sqrt(np.mean(np.square(npz["force"][0])))
        )


class TestReload:
    @pytest.mark.parametrize("change_rmax", [0, 1])
    @pytest.mark.parametrize("give_url", [True, False])
    @pytest.mark.parametrize("change_key_map", [True, False])
    def test_reload(self, npz_dataset, npz_data, change_rmax, give_url, change_key_map):
        r_max = npz_dataset.extra_fixed_fields["r_max"] + change_rmax
        keymap = npz_dataset.key_mapping.copy()  # the default one
        if change_key_map:
            keymap["x1"] = "x2"
        a = NpzDataset(
            file_name=npz_data,
            root=npz_dataset.root,
            extra_fixed_fields={"r_max": r_max},
            key_mapping=keymap,
            **({"url": "example.com/data.dat"} if give_url else {})
        )
        print(a.processed_file_names[0])
        print(npz_dataset.processed_file_names[0])
        assert (a.processed_dir == npz_dataset.processed_dir) == (
            (change_rmax == 0) and (not give_url) and (not change_key_map)
        )


class TestFromConfig:
    @pytest.mark.parametrize(
        "args",
        [
            dict(extra_fixed_fields={"r_max": 3.0}),
            dict(dataset_extra_fixed_fields={"r_max": 3.0}),
            dict(r_max=3.0),
            dict(r_max=3.0, extra_fixed_fields={}),
        ],
    )
    def test_npz(self, npz_data, root, args):
        config = Config(dict(dataset="npz", file_name=npz_data, root=root, **args))
        g = dataset_from_config(config)
        assert g.fixed_fields["r_max"] == 3
        assert isdir(g.root)
        assert isdir(g.processed_dir)
        assert isfile(g.processed_dir + "/data.pth")

    @pytest.mark.parametrize("prefix", ["dataset", "thingy"])
    def test_ase(self, ase_file, root, prefix):
        config = Config(
            dict(
                file_name=ase_file,
                root=root,
                extra_fixed_fields={"r_max": 3.0},
                ase_args=dict(format="extxyz"),
            )
        )
        config[prefix] = "ASEDataset"
        a = dataset_from_config(config, prefix=prefix)
        assert isdir(a.root)
        assert isdir(a.processed_dir)
        assert isfile(a.processed_dir + "/data.pth")

        # Test reload
        # Change some random ASE specific parameter
        # See https://wiki.fysik.dtu.dk/ase/ase/io/io.html
        config["ase_args"]["do_not_split_by_at_sign"] = True
        b = dataset_from_config(config, prefix=prefix)
        assert isdir(b.processed_dir)
        assert isfile(b.processed_dir + "/data.pth")
        assert a.processed_dir != b.processed_dir


class TestFromList:
    def test_from_atoms(self, molecules):
        dataset = ASEDataset.from_atoms_list(
            molecules, extra_fixed_fields={"r_max": 4.5}
        )
        assert len(dataset) == len(molecules)
        for i, mol in enumerate(molecules):
            assert np.array_equal(
                mol.get_atomic_numbers(), dataset.get(i).to_ase().get_atomic_numbers()
            )
