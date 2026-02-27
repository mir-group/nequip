import copy

import numpy as np
import torch

from nequip.data import (
    AtomicDataDict,
    from_dict,
    to_ase,
)


def test_to_ase_batches(atomic_batch):
    to_ase_atoms_batch = to_ase(atomic_batch)
    atomic_batch = AtomicDataDict.to_(atomic_batch, device="cpu")
    for batch_idx, atoms in enumerate(to_ase_atoms_batch):
        mask = atomic_batch[AtomicDataDict.BATCH_KEY] == batch_idx
        assert atoms.get_positions().shape == (len(atoms), 3)
        torch.testing.assert_close(
            torch.from_numpy(atoms.get_positions()),
            atomic_batch[AtomicDataDict.POSITIONS_KEY][mask],
        )
        assert atoms.get_atomic_numbers().shape == (len(atoms),)
        assert np.array_equal(
            atoms.get_atomic_numbers(),
            atomic_batch[AtomicDataDict.ATOMIC_NUMBERS_KEY][mask].view(-1),
        )

        if AtomicDataDict.CELL_KEY in atomic_batch:
            np.testing.assert_allclose(
                atoms.get_cell()[:],
                atomic_batch[AtomicDataDict.CELL_KEY][batch_idx].numpy(),
            )
        np.testing.assert_array_equal(
            atoms.get_pbc(),
            atomic_batch[AtomicDataDict.PBC_KEY][batch_idx].numpy(),
        )


def test_process_dict_invariance(CH3CHO):
    _, data = CH3CHO
    data1 = from_dict(data.copy())
    data2 = from_dict(data1.copy())
    for k in data.keys():
        torch.testing.assert_close(data1[k], data2[k])


def test_without_nodes(CH3CHO):
    # Non-periodic
    atoms, data = CH3CHO

    # add batches
    data = AtomicDataDict.with_batch_(data.copy())

    # === test with node indices ===
    which_nodes = [0, 5, 6]
    new_data = AtomicDataDict.without_nodes(data, which_nodes=which_nodes)
    assert AtomicDataDict.num_nodes(new_data) == len(atoms) - len(which_nodes)
    assert len(new_data[AtomicDataDict.BATCH_KEY]) == len(atoms) - len(which_nodes)
    assert new_data[AtomicDataDict.EDGE_INDEX_KEY].min() >= 0
    assert (
        new_data[AtomicDataDict.EDGE_INDEX_KEY].max()
        == AtomicDataDict.num_nodes(new_data) - 1
    )

    # === test with node mask ===
    which_nodes_mask = np.zeros(len(atoms), dtype=bool)
    which_nodes_mask[[0, 1, 2, 4]] = True
    new_data = AtomicDataDict.without_nodes(data, which_nodes=which_nodes_mask)
    assert AtomicDataDict.num_nodes(new_data) == len(atoms) - np.sum(which_nodes_mask)
    assert len(new_data[AtomicDataDict.BATCH_KEY]) == len(atoms) - np.sum(
        which_nodes_mask
    )
    assert new_data[AtomicDataDict.EDGE_INDEX_KEY].min() >= 0
    assert (
        new_data[AtomicDataDict.EDGE_INDEX_KEY].max()
        == AtomicDataDict.num_nodes(new_data) - 1
    )


def test_batching(CH3CHO):
    _, orig = CH3CHO
    N = 4

    # test unbatched vs batched
    data_list = []
    for _ in range(N):
        new = copy.deepcopy(orig)
        new[AtomicDataDict.POSITIONS_KEY] += torch.randn_like(
            new[AtomicDataDict.POSITIONS_KEY]
        )
        data_list.append(AtomicDataDict.with_batch_(new))
    batch = AtomicDataDict.batched_from_list(data_list)
    for i, orig in enumerate(data_list):
        new = AtomicDataDict.frame_from_batched(batch, i)
        for k, v in orig.items():
            assert torch.equal(v, new[k]), f"failed at iteration {i} for key {k}"

    # test batches of batched and unbatched vs unbatched
    data_list_add = [batch]
    for _ in range(N):
        new = copy.deepcopy(orig)
        new[AtomicDataDict.POSITIONS_KEY] += torch.randn_like(
            new[AtomicDataDict.POSITIONS_KEY]
        )
        data_list_add.append(AtomicDataDict.with_batch_(new))
    new_batch = AtomicDataDict.batched_from_list(data_list_add)
    combined_data_list = data_list + data_list_add[1:]
    for i, orig in enumerate(combined_data_list):
        new = AtomicDataDict.frame_from_batched(new_batch, i)
        for k, v in orig.items():
            assert torch.equal(v, new[k]), f"failed at iteration {i} for key {k}"
