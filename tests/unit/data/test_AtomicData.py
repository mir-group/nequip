import pytest
import copy

import numpy as np
import torch
from nequip.utils.torch_geometric import Batch

import ase.build
import ase.geometry
from ase.calculators.singlepoint import SinglePointCalculator

from nequip.data import AtomicData, AtomicDataDict
from nequip.data.AtomicData import neighbor_list_and_relative_vec

# skip_ase = pytest.mark.skipif(not has_ase, reason="ASE library is not installed")


def test_from_ase(CuFcc):
    atoms, data = CuFcc
    for key in [AtomicDataDict.FORCE_KEY, AtomicDataDict.POSITIONS_KEY]:
        assert data[key].shape == (len(atoms), 3)  # 4 species in this atoms


def test_to_ase(CH3CHO_no_typemap):
    atoms, data = CH3CHO_no_typemap
    to_ase_atoms = data.to_ase()
    assert np.allclose(atoms.get_positions(), to_ase_atoms.get_positions())
    assert np.array_equal(atoms.get_atomic_numbers(), to_ase_atoms.get_atomic_numbers())
    assert np.array_equal(atoms.get_pbc(), to_ase_atoms.get_pbc())
    assert np.array_equal(atoms.get_cell(), to_ase_atoms.get_cell())


def test_to_ase_batches(atomic_batch):
    atomic_data = AtomicData.from_dict(atomic_batch.to_dict())
    to_ase_atoms_batch = atomic_data.to_ase()
    for batch_idx, atoms in enumerate(to_ase_atoms_batch):
        mask = atomic_data.batch == batch_idx
        assert atoms.get_positions().shape == (len(atoms), 3)
        assert np.allclose(atoms.get_positions(), atomic_data.pos[mask])
        assert atoms.get_atomic_numbers().shape == (len(atoms),)
        assert np.array_equal(
            atoms.get_atomic_numbers(), atomic_data[AtomicDataDict.ATOM_TYPE_KEY][mask]
        )
        assert np.array_equal(atoms.get_cell(), atomic_data.cell[batch_idx])
        assert np.array_equal(atoms.get_pbc(), atomic_data.pbc[batch_idx])


def test_ase_roundtrip(CuFcc):
    atoms, data = CuFcc
    atoms2 = data.to_ase()
    assert np.allclose(atoms.get_positions(), atoms2.get_positions())
    assert np.array_equal(atoms.get_atomic_numbers(), atoms2.get_atomic_numbers())
    assert np.array_equal(atoms.get_pbc(), atoms2.get_pbc())
    assert np.allclose(atoms.get_cell(), atoms2.get_cell())
    assert np.allclose(atoms.calc.results["forces"], atoms2.calc.results["forces"])


def test_non_periodic_edge(CH3CHO):
    atoms, data = CH3CHO
    # check edges
    for edge in range(data.num_edges):
        real_displacement = (
            atoms.positions[data.edge_index[1, edge]]
            - atoms.positions[data.edge_index[0, edge]]
        )
        assert torch.allclose(
            data.get_edge_vectors()[edge],
            torch.as_tensor(real_displacement, dtype=torch.get_default_dtype()),
        )


def test_edges_missing():
    with pytest.raises(ValueError):
        # Check that when the cutoff is too small, the code complains
        # about a lack of edges in the graph.
        atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
        _ = AtomicData.from_ase(atoms, r_max=2.5)


def test_periodic_edge():
    atoms = ase.build.bulk("Cu", "fcc")
    dist = np.linalg.norm(atoms.cell[0])
    data = AtomicData.from_ase(atoms, r_max=1.05 * dist)
    edge_vecs = data.get_edge_vectors()
    assert edge_vecs.shape == (12, 3)  # 12 neighbors in close-packed bulk
    assert torch.allclose(
        edge_vecs.norm(dim=-1), torch.as_tensor(dist, dtype=torch.get_default_dtype())
    )


def test_without_nodes(CH3CHO):
    # Non-periodic
    atoms, data = CH3CHO
    which_nodes = [0, 5, 6]
    new_data = data.without_nodes(which_nodes=which_nodes)
    assert new_data.num_nodes == len(atoms) - len(which_nodes)
    new_data.debug()
    assert new_data.edge_index.min() >= 0
    assert new_data.edge_index.max() == new_data.num_nodes - 1

    which_nodes_mask = np.zeros(len(atoms), dtype=bool)
    which_nodes_mask[[0, 1, 2, 4]] = True
    new_data = data.without_nodes(which_nodes=which_nodes_mask)
    assert new_data.num_nodes == len(atoms) - np.sum(which_nodes_mask)
    new_data.debug()
    assert new_data.edge_index.min() >= 0
    assert new_data.edge_index.max() == new_data.num_nodes - 1


@pytest.mark.parametrize("periodic", [True, False])
def test_positions_grad(periodic, CH3CHO, CuFcc):

    if periodic:
        atoms, data = CuFcc
    else:
        atoms, data = CH3CHO

    data.pos.requires_grad_(True)
    assert data.get_edge_vectors().requires_grad

    torch.autograd.grad(data.get_edge_vectors().sum(), data.pos, create_graph=True)

    if periodic:
        # Test grad cell
        data.cell.requires_grad_(True)
        assert data.get_edge_vectors().requires_grad
        torch.autograd.grad(data.get_edge_vectors().sum(), data.cell, create_graph=True)


def test_some_periodic():
    # monolayer in xy,
    # only periodic in xy
    atoms = ase.build.fcc111("Al", size=(3, 3, 1), vacuum=0.0)
    assert all(atoms.pbc == (True, True, False))
    data = AtomicData.from_ase(atoms, r_max=2.9)  # first shell dist is 2.864A
    # Check number of neighbors:
    _, neighbor_count = np.unique(data.edge_index[0].numpy(), return_counts=True)
    assert (neighbor_count == 6).all()  # 6 neighbors
    # Check not periodic in z
    assert torch.allclose(
        data.get_edge_vectors()[:, 2],
        torch.zeros(data.num_edges, dtype=data.get_edge_vectors().dtype),
    )


def test_relative_vecs(H2):
    atoms, data = H2
    coords = data.pos
    edge_index_true = torch.LongTensor([[0, 1], [1, 0]])
    assert edge_index_set_equiv(data.edge_index, edge_index_true)
    assert torch.allclose(
        coords[1] - coords[0],
        data.get_edge_vectors()[(data.edge_index[0] == 0) & (data.edge_index[1] == 1)][
            0
        ],
    )
    assert torch.allclose(
        coords[0] - coords[1],
        data.get_edge_vectors()[(data.edge_index[0] == 1) & (data.edge_index[1] == 0)][
            0
        ],
    )


@pytest.mark.parametrize("interact", [True, False])
def test_self_interaction(interact, Si):
    points, _ = Si
    data = AtomicData.from_points(
        **points,
        self_interaction=interact,
    )
    if interact:
        true = torch.LongTensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    else:
        true = torch.LongTensor([[0, 1], [1, 0]])
    assert edge_index_set_equiv(data.edge_index, true)


def test_silicon_neighbors(Si):

    points, data = Si
    edge_index, cell_shifts, cell = neighbor_list_and_relative_vec(
        points["pos"],
        pbc=True,
        cell=points["cell"],
        r_max=points["r_max"],
        self_interaction=False,
    )
    edge_index_true = torch.LongTensor(
        [[0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]]
    )
    assert edge_index_set_equiv(edge_index, edge_index_true)
    assert edge_index_set_equiv(data.edge_index, edge_index_true)


def test_batching(Si):
    _, orig = Si
    N = 4
    datas = []
    for _ in range(N):
        new = copy.deepcopy(orig)
        new.pos += torch.randn_like(new.pos)
        datas.append(new)
    batch = Batch.from_data_list(datas)
    for i, orig in enumerate(datas):
        new = batch.get_example(i)
        for k, v in orig:
            assert torch.equal(v, new[k])


def edge_index_set_equiv(a, b):
    """Compare edge_index arrays in an unordered way."""
    # [[0, 1], [1, 0]] -> {(0, 1), (1, 0)}
    a = (
        a.numpy()
    )  # numpy gives ints when iterated, tensor gives non-identical scalar tensors.
    b = b.numpy()
    return set(zip(a[0], a[1])) == set(zip(b[0], b[1]))


@pytest.fixture(scope="session")
def H2(float_tolerance):
    atoms = ase.build.molecule("H2")
    data = AtomicData.from_ase(atoms, r_max=2.0)
    return atoms, data


@pytest.fixture(scope="session")
def CuFcc(float_tolerance):
    atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = SinglePointCalculator(
        atoms, **{"forces": np.random.random((len(atoms), 3))}
    )
    data = AtomicData.from_ase(atoms, r_max=4.0)
    return atoms, data


@pytest.fixture(scope="session")
def Si(float_tolerance):
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
        r_max=r_max,
        cell=lattice,
        pbc=True,
    )
    data = AtomicData.from_points(**points)
    return points, data
