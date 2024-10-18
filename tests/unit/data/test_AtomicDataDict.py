import pytest
import copy

import numpy as np
import torch
from ase import Atoms
import ase.build
import ase.geometry

from ase.calculators.singlepoint import SinglePointCalculator

from nequip.data import AtomicDataDict
from nequip.data._nl import neighbor_list_and_relative_vec
from nequip.utils.test import compare_neighborlists


def test_from_ase(CuFcc):
    atoms, data = CuFcc
    for key in [AtomicDataDict.FORCE_KEY, AtomicDataDict.POSITIONS_KEY]:
        assert data[key].shape == (len(atoms), 3)  # 4 species in this atoms


def test_to_ase(CH3CHO_no_typemap):
    atoms, data = CH3CHO_no_typemap
    to_ase_atoms = AtomicDataDict.to_ase(data)[0]
    assert np.allclose(atoms.get_positions(), to_ase_atoms.get_positions())
    assert np.array_equal(atoms.get_atomic_numbers(), to_ase_atoms.get_atomic_numbers())
    assert np.array_equal(atoms.get_pbc(), to_ase_atoms.get_pbc())
    assert np.array_equal(atoms.get_cell(), to_ase_atoms.get_cell())


def test_to_ase_batches(atomic_batch):
    to_ase_atoms_batch = AtomicDataDict.to_ase(atomic_batch)
    atomic_batch = AtomicDataDict.to_(atomic_batch, device="cpu")
    for batch_idx, atoms in enumerate(to_ase_atoms_batch):
        mask = atomic_batch["batch"] == batch_idx
        assert atoms.get_positions().shape == (len(atoms), 3)
        assert np.allclose(atoms.get_positions(), atomic_batch["pos"][mask])
        assert atoms.get_atomic_numbers().shape == (len(atoms),)
        assert np.array_equal(
            atoms.get_atomic_numbers(),
            atomic_batch[AtomicDataDict.ATOMIC_NUMBERS_KEY][mask].view(-1),
        )

        assert (
            np.max(
                np.abs(atoms.get_cell()[:] - atomic_batch["cell"][batch_idx].numpy())
            )
            == 0
        )
        assert not np.logical_xor(
            atoms.get_pbc(), atomic_batch["pbc"][batch_idx].numpy()
        ).all()


def test_ase_roundtrip(CuFcc):
    atoms, data = CuFcc
    atoms2 = AtomicDataDict.to_ase(data)[0]
    assert np.allclose(atoms.get_positions(), atoms2.get_positions())
    assert np.array_equal(atoms.get_atomic_numbers(), atoms2.get_atomic_numbers())
    assert np.array_equal(atoms.get_pbc(), atoms2.get_pbc())
    assert np.allclose(atoms.get_cell(), atoms2.get_cell())
    assert np.allclose(atoms.calc.results["forces"], atoms2.calc.results["forces"])


def test_process_dict_invariance(H2, CuFcc, CH3CHO):

    for system in [H2, CuFcc, CH3CHO]:
        atoms, data = system
        data1 = AtomicDataDict.from_dict(data.copy())
        data2 = AtomicDataDict.from_dict(data1.copy())
    for k in data.keys():
        assert torch.allclose(data1[k], data2[k])


def test_non_periodic_edge(CH3CHO):
    atoms, data = CH3CHO
    # check edges
    for edge in range(AtomicDataDict.num_edges(data)):
        real_displacement = (
            atoms.positions[data["edge_index"][1, edge]]
            - atoms.positions[data["edge_index"][0, edge]]
        )
        assert torch.allclose(
            AtomicDataDict.with_edge_vectors(data)["edge_vectors"][edge],
            torch.as_tensor(real_displacement, dtype=torch.get_default_dtype()),
        )


def test_periodic_edge():
    atoms = ase.build.bulk("Cu", "fcc")
    dist = np.linalg.norm(atoms.cell[0])
    data = AtomicDataDict.compute_neighborlist_(
        AtomicDataDict.from_ase(atoms), r_max=1.05 * dist
    )
    edge_vecs = AtomicDataDict.with_edge_vectors(data)["edge_vectors"]
    assert edge_vecs.shape == (12, 3)  # 12 neighbors in close-packed bulk
    assert torch.allclose(
        edge_vecs.norm(dim=-1), torch.as_tensor(dist, dtype=torch.get_default_dtype())
    )


def test_without_nodes(CH3CHO):
    # Non-periodic
    atoms, data = CH3CHO
    which_nodes = [0, 5, 6]
    new_data = AtomicDataDict.without_nodes(data, which_nodes=which_nodes)
    assert AtomicDataDict.num_nodes(new_data) == len(atoms) - len(which_nodes)
    assert new_data["edge_index"].min() >= 0
    assert new_data["edge_index"].max() == AtomicDataDict.num_nodes(new_data) - 1

    which_nodes_mask = np.zeros(len(atoms), dtype=bool)
    which_nodes_mask[[0, 1, 2, 4]] = True
    new_data = AtomicDataDict.without_nodes(data, which_nodes=which_nodes_mask)
    assert AtomicDataDict.num_nodes(new_data) == len(atoms) - np.sum(which_nodes_mask)
    assert new_data["edge_index"].min() >= 0
    assert new_data["edge_index"].max() == AtomicDataDict.num_nodes(new_data) - 1


@pytest.mark.parametrize("periodic", [True, False])
def test_positions_grad(periodic, H2, CuFcc):

    if periodic:
        atoms, data = CuFcc
    else:
        atoms, data = H2

    data["pos"].requires_grad_(True)
    data = AtomicDataDict.with_edge_vectors(data)
    assert data[AtomicDataDict.EDGE_VECTORS_KEY].requires_grad
    torch.autograd.grad(
        data[AtomicDataDict.EDGE_VECTORS_KEY].sum(),
        data["pos"],
        create_graph=True,
    )
    data.pop(AtomicDataDict.EDGE_VECTORS_KEY)

    if periodic:
        # Test grad cell
        data["cell"].requires_grad_(True)
        data = AtomicDataDict.with_edge_vectors(data)
        assert data[AtomicDataDict.EDGE_VECTORS_KEY].requires_grad
        torch.autograd.grad(
            data[AtomicDataDict.EDGE_VECTORS_KEY].sum(),
            data["cell"],
            create_graph=True,
        )


def test_some_periodic():
    # monolayer in xy,
    # only periodic in xy
    atoms = ase.build.fcc111("Al", size=(3, 3, 1), vacuum=0.0)
    assert all(atoms.pbc == (True, True, False))
    data = AtomicDataDict.compute_neighborlist_(
        AtomicDataDict.from_ase(atoms), r_max=2.9
    )  # first shell dist is 2.864 A
    # Check number of neighbors:
    _, neighbor_count = np.unique(data["edge_index"][0].numpy(), return_counts=True)
    assert (neighbor_count == 6).all()  # 6 neighbors
    # Check not periodic in z
    assert torch.allclose(
        AtomicDataDict.with_edge_vectors(data)["edge_vectors"][:, 2],
        torch.zeros(
            AtomicDataDict.num_edges(data),
            dtype=AtomicDataDict.with_edge_vectors(data)["edge_vectors"].dtype,
        ),
    )


def test_relative_vecs(H2):
    atoms, data = H2
    coords = data["pos"]
    edge_index_true = torch.LongTensor([[0, 1], [1, 0]])
    assert edge_index_set_equiv(data["edge_index"], edge_index_true)
    assert torch.allclose(
        coords[1] - coords[0],
        AtomicDataDict.with_edge_vectors(data)["edge_vectors"][
            (data["edge_index"][0] == 0) & (data["edge_index"][1] == 1)
        ][0],
    )
    assert torch.allclose(
        coords[0] - coords[1],
        AtomicDataDict.with_edge_vectors(data)["edge_vectors"][
            (data["edge_index"][0] == 1) & (data["edge_index"][1] == 0)
        ][0],
    )


def test_silicon_neighbors(Si):
    r_max, points, data = Si
    edge_index, cell_shifts, cell = neighbor_list_and_relative_vec(
        points["pos"],
        pbc=True,
        cell=points["cell"],
        r_max=r_max,
    )
    edge_index_true = torch.LongTensor(
        [[0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]]
    )
    assert edge_index_set_equiv(edge_index, edge_index_true)
    assert edge_index_set_equiv(data["edge_index"], edge_index_true)


@pytest.mark.parametrize("alt_nl_method", ["matscipy", "vesin"])
def test_neighborlist_consistency(alt_nl_method, CH3CHO, CuFcc, Si):

    # check if modules are installed
    try:
        if alt_nl_method == "vesin":
            import vesin  # noqa: F401
    except ImportError:
        pytest.skip(f"package for {alt_nl_method} neighborlist not available")

    CH3CHO_atoms, _ = CH3CHO
    CuFcc_atoms, _ = CuFcc
    _, Si_points, _ = Si
    r_max = 4.0

    Si_data = AtomicDataDict.from_dict(Si_points)
    for atoms_or_data in [CH3CHO_atoms, CuFcc_atoms, Si_data]:
        compare_neighborlists(atoms_or_data, nl1="ase", nl2=alt_nl_method, r_max=r_max)


@pytest.mark.parametrize("nl_method", ["ase", "matscipy", "vesin"])
def test_no_neighbors(nl_method):
    """Tests that the neighborlist is empty if there are no neighbors."""
    # check if modules are installed
    try:
        if nl_method == "vesin":
            import vesin  # noqa: F401
    except ImportError:
        pytest.skip(f"package for {nl_method} neighborlist not available")

    # isolated atom
    H = Atoms("H", positions=[[0, 0, 0]], cell=20 * np.eye(3))
    data = AtomicDataDict.compute_neighborlist_(AtomicDataDict.from_ase(H), r_max=2.5)
    assert data[AtomicDataDict.EDGE_INDEX_KEY].numel() == 0
    assert data[AtomicDataDict.EDGE_CELL_SHIFT_KEY].numel() == 0

    # cutoff smaller than interatomic distance
    Cu = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
    data = AtomicDataDict.compute_neighborlist_(AtomicDataDict.from_ase(Cu), r_max=2.5)
    assert data[AtomicDataDict.EDGE_INDEX_KEY].numel() == 0
    assert data[AtomicDataDict.EDGE_CELL_SHIFT_KEY].numel() == 0


def test_batching(Si):
    _, _, orig = Si
    N = 4

    # test unbatched vs batched
    data_list = []
    for _ in range(N):
        new = copy.deepcopy(orig)
        new["pos"] += torch.randn_like(new["pos"])
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
        new["pos"] += torch.randn_like(new["pos"])
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
    data = AtomicDataDict.compute_neighborlist_(
        AtomicDataDict.from_ase(atoms),
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
    data = AtomicDataDict.compute_neighborlist_(
        AtomicDataDict.from_ase(atoms),
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
    data = AtomicDataDict.compute_neighborlist_(
        AtomicDataDict.from_dict(points),
        r_max=r_max,
        NL="ase",
    )
    return r_max, points, data
