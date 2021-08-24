import numpy as np
import pytest
import tempfile

from torch.utils.data import SubsetRandomSampler

from nequip.data import NpzDataset, DataLoader


class TestInit:
    def test_init(self, npz_dataset):
        dl = DataLoader(
            npz_dataset, batch_size=2, shuffle=True, exclude_keys=["energy"]
        )

    def test_subset(self, npz_dataset):
        subset = npz_dataset[[1, 3]]


class TestLoop:
    def test_whole(self, dloader):
        for i, batch in enumerate(dloader):
            print(i)
            print(batch)

    def test_non_divisor(self, npz_dataset):
        dataset = [npz_dataset[i] for i in range(7)]  # make it odd length
        dl = DataLoader(dataset, batch_size=2, shuffle=True, exclude_keys=["energy"])
        dl_iter = iter(dl)
        for _ in range(3):
            batch = next(dl_iter)
            assert batch.num_graphs == 2
        last_batch = next(dl_iter)
        assert last_batch.num_graphs == 1
        assert last_batch.batch.max() == 0
        with pytest.raises(StopIteration):
            next(dl_iter)

    def test_subset(self, npz_dataset):
        dloader = DataLoader(
            npz_dataset[:4], batch_size=2, shuffle=True, exclude_keys=["energy"]
        )
        for i, batch in enumerate(dloader):
            print(i)
            print(batch)

    def test_subset_sampler(self, npz_dataset):
        dloader = DataLoader(
            npz_dataset,
            batch_size=2,
            sampler=SubsetRandomSampler(indices=[0, 1, 2, 3, 4]),
            exclude_keys=["energy"],
        )
        for i, batch in enumerate(dloader):
            print(i)
            print(batch)


@pytest.fixture(scope="module")
def npz_dataset():
    natoms = 3
    nframes = 8
    npz = dict(
        positions=np.random.random((nframes, natoms, 3)),
        force=np.random.random((nframes, natoms, 3)),
        energy=np.random.random(nframes),
        Z=np.random.randint(1, 108, size=(nframes, natoms)),
    )
    with tempfile.TemporaryDirectory() as folder:
        np.savez(folder + "/npzdata.npz", **npz)
        a = NpzDataset(
            file_name=folder + "/npzdata.npz",
            root=folder,
            extra_fixed_fields={"r_max": 3},
        )
        yield a


@pytest.fixture(scope="class")
def dloader(npz_dataset):
    yield DataLoader(npz_dataset, batch_size=2, shuffle=True, exclude_keys=["energy"])
