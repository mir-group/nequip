import torch
import numpy as np

from ase import Atoms
import ase.build

from nequip.data import AtomicDataDict, from_dict, from_ase
from nequip.data.transforms import (
    VirialToStressTransform,
    StressSignFlipTransform,
    NeighborListTransform,
    SortedNeighborListTransform,
)


def test_VirialToStressTransform():
    # create data
    num_frames = 3
    num_atoms = 17
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(num_atoms, 3),
        AtomicDataDict.CELL_KEY: torch.randn(num_frames, 3, 3),
        AtomicDataDict.PBC_KEY: torch.full((num_frames, 3), True),
        AtomicDataDict.VIRIAL_KEY: torch.randn(num_frames, 3, 3),
        AtomicDataDict.NUM_NODES_KEY: torch.tensor([4, 6, 7]),
    }
    # not strictly needed, but useful to test that `from_dict` runs with the given input dict
    data = from_dict(data)
    transformed = VirialToStressTransform()(data)
    # implementation is trivial, good enough to test that it runs and has the correct shape
    assert AtomicDataDict.STRESS_KEY in transformed
    assert transformed[AtomicDataDict.STRESS_KEY].shape == (num_frames, 3, 3)


def test_StressSignFlipTransform():
    # create data
    num_frames = 3
    num_atoms = 17
    stress = torch.randn(num_frames, 3, 3)
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(num_atoms, 3),
        AtomicDataDict.NUM_NODES_KEY: torch.tensor([4, 6, 7]),
        AtomicDataDict.PBC_KEY: torch.full((num_frames, 3), True),
        AtomicDataDict.STRESS_KEY: stress.clone(),
    }
    data = from_dict(data)
    result = StressSignFlipTransform()(data)[AtomicDataDict.STRESS_KEY]
    assert result.shape == (num_frames, 3, 3)
    assert torch.allclose(result, -stress)


def test_neighborlist_basic():
    """Test basic neighborlist functionality."""
    # create a system for testing
    atoms = ase.build.bulk("Cu", "fcc", a=3.6)
    atoms = ase.build.make_supercell(atoms, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    data_base = from_ase(atoms)

    # test basic transform
    data_basic = NeighborListTransform(r_max=4.0)(data_base.copy())

    # check basic transform outputs
    assert AtomicDataDict.EDGE_INDEX_KEY in data_basic
    assert AtomicDataDict.EDGE_TRANSPOSE_PERM_KEY not in data_basic

    # test empty case
    atoms_empty = Atoms("H", positions=[[0, 0, 0]], cell=20 * np.eye(3))
    data_empty = from_ase(atoms_empty)
    basic_result = NeighborListTransform(r_max=2.5)(data_empty)
    assert basic_result[AtomicDataDict.EDGE_INDEX_KEY].numel() == 0
    assert basic_result[AtomicDataDict.EDGE_CELL_SHIFT_KEY].numel() == 0


def test_sorted_neighborlist_with_permutation():
    """Test SortedNeighborListTransform with sorting, inheritance, and transpose permutation."""
    # create a system with known structure
    atoms = ase.build.bulk("Cu", "fcc", a=3.6)
    atoms = ase.build.make_supercell(atoms, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    atoms.rattle()

    data_base = from_ase(atoms)

    # test inheritance and basic properties
    sorted_transform = SortedNeighborListTransform(r_max=4.0)
    basic_transform = NeighborListTransform(r_max=4.0)
    assert isinstance(sorted_transform, NeighborListTransform)
    assert type(sorted_transform) is not type(basic_transform)

    # apply sorted transform
    data_sorted = sorted_transform(data_base.copy())
    data_basic = basic_transform(data_base.copy())

    # check that sorted transform has all expected keys
    assert AtomicDataDict.EDGE_INDEX_KEY in data_sorted
    assert AtomicDataDict.EDGE_TRANSPOSE_PERM_KEY in data_sorted

    # check that basic transform doesn't have sorted keys
    assert AtomicDataDict.EDGE_TRANSPOSE_PERM_KEY not in data_basic

    sorted_edge_index = data_sorted[AtomicDataDict.EDGE_INDEX_KEY]
    transpose_perm = data_sorted[AtomicDataDict.EDGE_TRANSPOSE_PERM_KEY]

    # verify that both transforms produce same edges
    edge_index_basic = data_basic[AtomicDataDict.EDGE_INDEX_KEY]
    edges_basic = set(zip(edge_index_basic[0].tolist(), edge_index_basic[1].tolist()))
    edges_sorted = set(
        zip(sorted_edge_index[0].tolist(), sorted_edge_index[1].tolist())
    )
    assert (
        edges_basic == edges_sorted
    ), "Basic and sorted should contain identical edges"

    # check permutation properties
    assert transpose_perm.shape == (
        sorted_edge_index.size(1),
    ), "Transpose permutation should have same length as edges"
    assert (
        transpose_perm.dtype == torch.long
    ), "Transpose permutation should be long tensor"

    # verify permutation is valid
    sorted_perm = torch.sort(transpose_perm)[0]
    expected_indices = torch.arange(sorted_edge_index.size(1))
    assert torch.equal(
        sorted_perm, expected_indices
    ), "Transpose permutation should be valid"

    receivers = sorted_edge_index[0]
    senders = sorted_edge_index[1]

    # verify row-major ordering (sorted by receiver first, then sender)
    assert torch.equal(
        receivers, torch.sort(receivers)[0]
    ), "Receivers should be sorted"
    unique_receivers = torch.unique(receivers)
    for receiver in unique_receivers:
        mask = receivers == receiver
        senders_for_receiver = senders[mask]
        assert torch.equal(
            senders_for_receiver, torch.sort(senders_for_receiver)[0]
        ), f"Senders not sorted for receiver {receiver}"

    # apply transpose permutation and verify column-major ordering
    col_major_edge_index = torch.index_select(sorted_edge_index, 1, transpose_perm)
    col_major_receivers = col_major_edge_index[0]
    col_major_senders = col_major_edge_index[1]

    # verify full column-major ordering (sorted by sender first, then receiver)
    assert torch.equal(
        col_major_senders, torch.sort(col_major_senders)[0]
    ), "Senders should be sorted in column-major"
    unique_senders = torch.unique(col_major_senders)
    for sender in unique_senders:
        mask = col_major_senders == sender
        receivers_for_sender = col_major_receivers[mask]
        assert torch.equal(
            receivers_for_sender, torch.sort(receivers_for_sender)[0]
        ), f"Receivers not sorted for sender {sender}"

    # verify same edges in both orderings
    row_major_edges = set(zip(receivers.tolist(), senders.tolist()))
    col_major_edges = set(zip(col_major_receivers.tolist(), col_major_senders.tolist()))
    assert (
        row_major_edges == col_major_edges
    ), "Row-major and column-major should contain identical edges"

    # test empty case
    atoms_empty = Atoms("H", positions=[[0, 0, 0]], cell=20 * np.eye(3))
    data_empty = from_ase(atoms_empty)
    sorted_result = SortedNeighborListTransform(r_max=2.5)(data_empty)
    assert sorted_result[AtomicDataDict.EDGE_INDEX_KEY].numel() == 0
    assert AtomicDataDict.EDGE_TRANSPOSE_PERM_KEY not in sorted_result
