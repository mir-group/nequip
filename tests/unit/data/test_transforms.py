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
    assert edges_basic == edges_sorted, (
        "Basic and sorted should contain identical edges"
    )

    # check permutation properties
    assert transpose_perm.shape == (sorted_edge_index.size(1),), (
        "Transpose permutation should have same length as edges"
    )
    assert transpose_perm.dtype == torch.long, (
        "Transpose permutation should be long tensor"
    )

    # verify permutation is valid
    sorted_perm = torch.sort(transpose_perm)[0]
    expected_indices = torch.arange(sorted_edge_index.size(1))
    assert torch.equal(sorted_perm, expected_indices), (
        "Transpose permutation should be valid"
    )

    receivers = sorted_edge_index[0]
    senders = sorted_edge_index[1]

    # verify row-major ordering (sorted by receiver first, then sender)
    assert torch.equal(receivers, torch.sort(receivers)[0]), (
        "Receivers should be sorted"
    )
    unique_receivers = torch.unique(receivers)
    for receiver in unique_receivers:
        mask = receivers == receiver
        senders_for_receiver = senders[mask]
        assert torch.equal(senders_for_receiver, torch.sort(senders_for_receiver)[0]), (
            f"Senders not sorted for receiver {receiver}"
        )

    # apply transpose permutation and verify column-major ordering
    col_major_edge_index = torch.index_select(sorted_edge_index, 1, transpose_perm)
    col_major_receivers = col_major_edge_index[0]
    col_major_senders = col_major_edge_index[1]

    # verify full column-major ordering (sorted by sender first, then receiver)
    assert torch.equal(col_major_senders, torch.sort(col_major_senders)[0]), (
        "Senders should be sorted in column-major"
    )
    unique_senders = torch.unique(col_major_senders)
    for sender in unique_senders:
        mask = col_major_senders == sender
        receivers_for_sender = col_major_receivers[mask]
        assert torch.equal(receivers_for_sender, torch.sort(receivers_for_sender)[0]), (
            f"Receivers not sorted for sender {sender}"
        )

    # verify same edges in both orderings
    row_major_edges = set(zip(receivers.tolist(), senders.tolist()))
    col_major_edges = set(zip(col_major_receivers.tolist(), col_major_senders.tolist()))
    assert row_major_edges == col_major_edges, (
        "Row-major and column-major should contain identical edges"
    )

    # test empty case
    atoms_empty = Atoms("H", positions=[[0, 0, 0]], cell=20 * np.eye(3))
    data_empty = from_ase(atoms_empty)
    sorted_result = SortedNeighborListTransform(r_max=2.5)(data_empty)
    assert sorted_result[AtomicDataDict.EDGE_INDEX_KEY].numel() == 0
    assert AtomicDataDict.EDGE_TRANSPOSE_PERM_KEY not in sorted_result


def test_per_edge_type_cutoff_basic():
    """Test basic per-edge-type cutoff functionality."""
    # create a simple two-atom system
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 1]),  # H and C
        AtomicDataDict.CELL_KEY: torch.eye(3) * 10,
        AtomicDataDict.PBC_KEY: torch.tensor([True, True, True]),
    }
    data = from_dict(data)

    type_names = ["H", "C"]
    r_max = 3.0

    # test uniform cutoff (should behave like normal neighborlist)
    per_edge_type_cutoff = {"H": 2.5, "C": 2.5}
    transform = NeighborListTransform(
        r_max=r_max, per_edge_type_cutoff=per_edge_type_cutoff, type_names=type_names
    )
    result = transform(data.copy())

    # should have edges since distance (2.0) < cutoff (2.5)
    assert result[AtomicDataDict.EDGE_INDEX_KEY].shape[1] > 0

    # check that no extra keys are left behind from EdgeLengthNormalizer
    assert AtomicDataDict.EDGE_LENGTH_KEY not in result
    assert AtomicDataDict.EDGE_VECTORS_KEY not in result
    assert AtomicDataDict.NORM_LENGTH_KEY not in result
    assert AtomicDataDict.EDGE_TYPE_KEY not in result

    # test with smaller cutoff that excludes the edge
    per_edge_type_cutoff_small = {"H": 1.5, "C": 1.5}
    transform_small = NeighborListTransform(
        r_max=r_max,
        per_edge_type_cutoff=per_edge_type_cutoff_small,
        type_names=type_names,
    )
    result_small = transform_small(data.copy())

    # should have no edges since distance (2.0) > cutoff (1.5)
    assert result_small[AtomicDataDict.EDGE_INDEX_KEY].shape[1] == 0

    # check that no extra keys are left behind
    assert AtomicDataDict.EDGE_LENGTH_KEY not in result_small
    assert AtomicDataDict.EDGE_VECTORS_KEY not in result_small
    assert AtomicDataDict.NORM_LENGTH_KEY not in result_small
    assert AtomicDataDict.EDGE_TYPE_KEY not in result_small


def test_per_edge_type_cutoff_asymmetric():
    """Test asymmetric per-edge-type cutoffs (different cutoffs for different pairs)."""
    # create a three-atom system: H-C-O chain
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]  # H  # C  # O
        ),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 1, 2]),  # H, C, O
        AtomicDataDict.CELL_KEY: torch.eye(3) * 10,
        AtomicDataDict.PBC_KEY: torch.tensor([True, True, True]),
    }
    data = from_dict(data)

    type_names = ["H", "C", "O"]
    r_max = 4.0

    # asymmetric cutoffs: H-C allowed (1.5 < 2.0), H-O not allowed (3.0 > 1.0)
    per_edge_type_cutoff = {
        "H": {"H": 2.0, "C": 2.0, "O": 1.0},
        "C": {"H": 2.0, "C": 2.0, "O": 1.8},
        "O": {"H": 1.0, "C": 1.8, "O": 2.0},
    }

    transform = NeighborListTransform(
        r_max=r_max, per_edge_type_cutoff=per_edge_type_cutoff, type_names=type_names
    )
    result = transform(data.copy())

    edge_index = result[AtomicDataDict.EDGE_INDEX_KEY]

    # convert to edge list for easier checking
    edges = [
        (edge_index[0, i].item(), edge_index[1, i].item())
        for i in range(edge_index.shape[1])
    ]

    # should have H-C edges (distance 1.5 < cutoff 2.0)
    assert (0, 1) in edges or (1, 0) in edges

    # should NOT have H-O edges (distance 3.0 > cutoff 1.0)
    h_o_present = (0, 2) in edges or (2, 0) in edges
    assert not h_o_present, "H-O edge should not be present due to cutoff"


def test_per_edge_type_cutoff_nacl():
    """Test per-edge-type cutoffs on NaCl structure with asymmetric cutoffs."""
    # create NaCl structure
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.64)
    atoms = ase.build.make_supercell(atoms, [[2, 0, 0], [0, 2, 0], [0, 0, 1]])

    data = from_ase(atoms)
    # manually set atom types: Na=0, Cl=1
    data[AtomicDataDict.ATOM_TYPE_KEY] = torch.tensor(
        [i % 2 for i in range(len(atoms))]
    )

    type_names = ["Na", "Cl"]
    r_max = 4.0

    # asymmetric cutoffs: allow Na-Cl but restrict Na-Na and Cl-Cl
    per_edge_type_cutoff = {
        "Na": {"Cl": 3.5, "Na": 2.0},  # Na-Cl allowed, Na-Na restricted
        "Cl": {"Na": 3.5, "Cl": 2.0},  # Cl-Na allowed, Cl-Cl restricted
    }

    # test without per-edge-type cutoffs
    basic_transform = NeighborListTransform(r_max=r_max)
    basic_result = basic_transform(data.copy())

    # test with per-edge-type cutoffs
    per_type_transform = NeighborListTransform(
        r_max=r_max, per_edge_type_cutoff=per_edge_type_cutoff, type_names=type_names
    )
    per_type_result = per_type_transform(data.copy())

    # per-edge-type should have fewer or equal edges
    assert (
        per_type_result[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        <= basic_result[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
    )

    # analyze edge types in per-edge-type result
    edge_index = per_type_result[AtomicDataDict.EDGE_INDEX_KEY]
    atom_types = data[AtomicDataDict.ATOM_TYPE_KEY]

    edge_types = []
    for i in range(edge_index.shape[1]):
        sender_type = atom_types[edge_index[0, i]].item()
        receiver_type = atom_types[edge_index[1, i]].item()
        edge_types.append((sender_type, receiver_type))

    # should have Na-Cl edges (type pair (0,1) and (1,0))
    na_cl_edges = sum(1 for t in edge_types if t in [(0, 1), (1, 0)])
    assert na_cl_edges > 0, "Should have Na-Cl edges"


def test_per_edge_type_cutoff_inheritance():
    """Test that SortedNeighborListTransform inherits per-edge-type cutoff functionality."""
    # create a simple system
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]]
        ),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 0, 1]),  # H, H, C
        AtomicDataDict.CELL_KEY: torch.eye(3) * 10,
        AtomicDataDict.PBC_KEY: torch.tensor([True, True, True]),
    }
    data = from_dict(data)

    type_names = ["H", "C"]
    r_max = 3.0
    per_edge_type_cutoff = {"H": 1.5, "C": 1.5}  # only allow first H-H pair

    # test basic transform
    basic_transform = NeighborListTransform(
        r_max=r_max, per_edge_type_cutoff=per_edge_type_cutoff, type_names=type_names
    )
    basic_result = basic_transform(data.copy())

    # test sorted transform
    sorted_transform = SortedNeighborListTransform(
        r_max=r_max, per_edge_type_cutoff=per_edge_type_cutoff, type_names=type_names
    )
    sorted_result = sorted_transform(data.copy())

    # both should have same number of edges (after per-edge-type pruning)
    assert (
        basic_result[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        == sorted_result[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
    )

    # sorted should have transpose permutation
    assert AtomicDataDict.EDGE_TRANSPOSE_PERM_KEY in sorted_result
    assert AtomicDataDict.EDGE_TRANSPOSE_PERM_KEY not in basic_result

    # check that same edges are present in both (though possibly in different order)
    basic_edges = set(
        zip(
            basic_result[AtomicDataDict.EDGE_INDEX_KEY][0].tolist(),
            basic_result[AtomicDataDict.EDGE_INDEX_KEY][1].tolist(),
        )
    )
    sorted_edges = set(
        zip(
            sorted_result[AtomicDataDict.EDGE_INDEX_KEY][0].tolist(),
            sorted_result[AtomicDataDict.EDGE_INDEX_KEY][1].tolist(),
        )
    )
    assert basic_edges == sorted_edges


def test_per_edge_type_cutoff_validation():
    """Test validation of per-edge-type cutoff parameters."""
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.tensor([[0.0, 0.0, 0.0]]),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0]),
        AtomicDataDict.CELL_KEY: torch.eye(3) * 10,
        AtomicDataDict.PBC_KEY: torch.tensor([True, True, True]),
    }
    data = from_dict(data)

    # should require type_names when per_edge_type_cutoff is provided
    try:
        _ = NeighborListTransform(
            r_max=3.0,
            per_edge_type_cutoff={"H": 2.0},
            type_names=None,  # missing type_names
        )
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_per_edge_type_cutoff_defaults():
    """Test that missing source/target types default to r_max."""
    # create a simple system
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.5, 0.0, 0.0]]
        ),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 1, 2]),  # H, C, O
        AtomicDataDict.CELL_KEY: torch.eye(3) * 10,
        AtomicDataDict.PBC_KEY: torch.tensor([True, True, True]),
    }
    data = from_dict(data)

    type_names = ["H", "C", "O"]
    r_max = 4.0

    # Test 1: Missing source types should default to r_max
    per_edge_type_cutoff = {"H": 2.0}  # C and O missing, should default to r_max=4.0
    transform = NeighborListTransform(
        r_max=r_max, per_edge_type_cutoff=per_edge_type_cutoff, type_names=type_names
    )
    result = transform(data.copy())

    # Should have edges for C-O (distance 2.5 < r_max=4.0) since C and O default to r_max
    edge_index = result[AtomicDataDict.EDGE_INDEX_KEY]
    edges = [
        (edge_index[0, i].item(), edge_index[1, i].item())
        for i in range(edge_index.shape[1])
    ]
    c_o_present = (1, 2) in edges or (2, 1) in edges
    assert c_o_present, "C-O edge should be present since C and O default to r_max=4.0"

    # Test 2: Missing target types should default to r_max
    per_edge_type_cutoff = {
        "H": {"C": 2.0},  # missing H->H and H->O, should default to r_max=4.0
        "C": {"H": 2.0, "C": 2.0, "O": 2.0},
        "O": {"H": 2.0, "C": 2.0, "O": 2.0},
    }
    transform = NeighborListTransform(
        r_max=r_max, per_edge_type_cutoff=per_edge_type_cutoff, type_names=type_names
    )
    result = transform(data.copy())

    # Should have H-O edge (distance 3.5 < r_max=4.0) since H->O defaults to r_max
    edge_index = result[AtomicDataDict.EDGE_INDEX_KEY]
    edges = [
        (edge_index[0, i].item(), edge_index[1, i].item())
        for i in range(edge_index.shape[1])
    ]
    h_o_present = (0, 2) in edges or (2, 0) in edges
    assert h_o_present, "H-O edge should be present since H->O defaults to r_max=4.0"
