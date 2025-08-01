import pytest

import torch
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset

from nequip.data import AtomicDataDict
from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper


def test_type_mapper():
    tm = ChemicalSpeciesToAtomTypeMapper(
        chemical_symbols=["C", "H"],
    )
    data = {AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.as_tensor([1, 1, 6, 1, 6, 6, 6])}
    data = tm(data)
    atom_types = data[AtomicDataDict.ATOM_TYPE_KEY]
    assert torch.all(atom_types == torch.as_tensor([1, 1, 0, 1, 0, 0, 0]))


class TestDataset:
    def test_subset(self, dataset):
        Subset(dataset, [1, 3])


class TestDataLoader:
    def test_whole(self, dataset):
        dloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=AtomicDataDict.batched_from_list,
        )

        for i, batch in enumerate(dloader):
            print(i)
            print(batch)

    def test_non_divisor(self, dataset):
        dataset = [dataset[i] for i in range(7)]  # make it odd length
        dl = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=AtomicDataDict.batched_from_list,
        )
        dl_iter = iter(dl)
        for _ in range(3):
            batch = next(dl_iter)
            assert AtomicDataDict.num_frames(batch) == 2
        last_batch = next(dl_iter)
        assert AtomicDataDict.num_frames(last_batch) == 1
        assert last_batch["batch"].max() == 0
        with pytest.raises(StopIteration):
            next(dl_iter)

    def test_subset(self, dataset):
        dloader = DataLoader(
            dataset[:4],
            batch_size=2,
            shuffle=True,
            collate_fn=AtomicDataDict.batched_from_list,
        )
        for i, batch in enumerate(dloader):
            print(i)
            print(batch)

    def test_subset_sampler(self, dataset):
        dloader = DataLoader(
            dataset,
            batch_size=2,
            sampler=SubsetRandomSampler(indices=[0, 1, 2, 3, 4]),
            collate_fn=AtomicDataDict.batched_from_list,
        )
        for i, batch in enumerate(dloader):
            print(i)
            print(batch)
