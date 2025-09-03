import pytest
import copy

import numpy as np
import torch
from ase import Atoms
import ase.build

from ase.calculators.singlepoint import SinglePointCalculator

from nequip.data import (
    AtomicDataDict,
    from_dict,
    from_ase,
    to_ase,
    compute_neighborlist_,
)
from nequip.data._nl import neighbor_list_and_relative_vec
from nequip.utils.test import compare_neighborlists

# check for optional neighborlist libraries
try:
    import vesin  # noqa: F401

    VESIN_AVAILABLE = True
except ImportError:
    VESIN_AVAILABLE = False

# build parametrize lists based on available libraries
ALT_NL_METHODS = ["matscipy"]
if VESIN_AVAILABLE:
    ALT_NL_METHODS.append("vesin")

NL_METHODS = ["ase", "matscipy"]
if VESIN_AVAILABLE:
    NL_METHODS.append("vesin")


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

        assert (
            np.max(
                np.abs(
                    atoms.get_cell()[:]
                    - atomic_batch[AtomicDataDict.CELL_KEY][batch_idx].numpy()
                )
            )
            == 0
        )
        assert not np.logical_xor(
            atoms.get_pbc(), atomic_batch[AtomicDataDict.PBC_KEY][batch_idx].numpy()
        ).all()


def test_process_dict_invariance(H2, CuFcc, CH3CHO):
    for system in [H2, CuFcc, CH3CHO]:
        _, data = system
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


def test_silicon_neighbors(Si):
    r_max, points, data = Si
    edge_index, cell_shifts, cell = neighbor_list_and_relative_vec(
        points[AtomicDataDict.POSITIONS_KEY],
        pbc=True,
        cell=points[AtomicDataDict.CELL_KEY],
        r_max=r_max,
    )
    edge_index_true = torch.LongTensor(
        [[0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]]
    )
    assert edge_index_set_equiv(edge_index, edge_index_true)
    assert edge_index_set_equiv(data[AtomicDataDict.EDGE_INDEX_KEY], edge_index_true)


@pytest.mark.parametrize("alt_nl_method", ALT_NL_METHODS)
def test_neighborlist_consistency(alt_nl_method, CH3CHO, CuFcc, Si):
    CH3CHO_atoms, _ = CH3CHO
    CuFcc_atoms, _ = CuFcc
    _, Si_points, _ = Si
    r_max = 4.0

    Si_data = from_dict(Si_points)
    for atoms_or_data in [CH3CHO_atoms, CuFcc_atoms, Si_data]:
        compare_neighborlists(atoms_or_data, nl1="ase", nl2=alt_nl_method, r_max=r_max)


@pytest.mark.parametrize("nl_method", NL_METHODS)
def test_no_neighbors(nl_method):
    """Tests that the neighborlist is empty if there are no neighbors."""

    # isolated atom
    H = Atoms("H", positions=[[0, 0, 0]], cell=20 * np.eye(3))
    data = compute_neighborlist_(from_ase(H), r_max=2.5, NL=nl_method)
    assert data[AtomicDataDict.EDGE_INDEX_KEY].numel() == 0
    assert data[AtomicDataDict.EDGE_CELL_SHIFT_KEY].numel() == 0

    # cutoff smaller than interatomic distance
    Cu = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
    data = compute_neighborlist_(from_ase(Cu), r_max=2.5, NL=nl_method)
    assert data[AtomicDataDict.EDGE_INDEX_KEY].numel() == 0
    assert data[AtomicDataDict.EDGE_CELL_SHIFT_KEY].numel() == 0


def test_batching(Si):
    _, _, orig = Si
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


def edge_index_set_equiv(a, b):
    """Compare edge_index arrays in an unordered way."""
    # [[0, 1], [1, 0]] -> {(0, 1), (1, 0)}
    a = (
        a.numpy()
    )  # numpy gives ints when iterated, tensor gives non-identical scalar tensors.
    b = b.numpy()
    return set(zip(a[0], a[1])) == set(zip(b[0], b[1]))


@pytest.fixture(scope="function")
def H2():
    atoms = ase.build.molecule("H2")
    data = compute_neighborlist_(
        from_ase(atoms),
        r_max=2.0,
        NL="ase",
    )
    return atoms, data


@pytest.fixture(scope="function")
def CuFcc():
    atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = SinglePointCalculator(
        atoms, **{"forces": np.random.random((len(atoms), 3))}
    )
    data = compute_neighborlist_(
        from_ase(atoms),
        r_max=4.0,
        NL="ase",
    )
    return atoms, data


@pytest.fixture(scope="function")
def Si():
    lattice = torch.tensor(
        [
            [3.34939851, 0, 1.93377613],
            [1.11646617, 3.1578432, 1.93377613],
            [0, 0, 3.86755226],
        ]
    )
    coords = torch.tensor([[0, 0, 0], [1.11646617, 0.7894608, 1.93377613]])
    r_max = 2.5
    points = dict(
        # z=torch.zeros(size=(len(coords), 1)),
        pos=coords,
        cell=lattice,
        pbc=True,
    )
    data = compute_neighborlist_(
        from_dict(points),
        r_max=r_max,
        NL="ase",
    )
    return r_max, points, data
