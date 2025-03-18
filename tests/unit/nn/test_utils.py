import pytest

import numpy as np
import torch

from nequip.data import AtomicDataDict, compute_neighborlist_, from_ase
from nequip.nn.embedding import NodeTypeEmbed
from nequip.nn.embedding._edge import _process_per_edge_type_cutoff
from nequip.nn import SequentialGraphNetwork, SaveForOutput, AtomwiseLinear, GraphModel
from nequip.nn.utils import with_edge_vectors_
from nequip.utils import dtype_from_name, torch_default_dtype
from nequip.utils.global_dtype import _GLOBAL_DTYPE

import ase


def test_process_per_edge_cutoff():
    # Unit case
    assert torch.all(
        _process_per_edge_type_cutoff(["O"], {"O": 2.0}, r_max=4.0)
        == torch.as_tensor([[2.0]], dtype=_GLOBAL_DTYPE)
    )
    # Unit case more complex
    assert torch.all(
        _process_per_edge_type_cutoff(["O"], {"O": {"O": 2.0}}, r_max=4.0)
        == torch.as_tensor([[2.0]], dtype=_GLOBAL_DTYPE)
    )

    # Normal behavior
    type_names = ["H", "C", "O"]
    per_edge_type_cutoff = {"H": 2.0, "C": {"H": 4.0, "C": 3.5, "O": 3.7}, "O": 3.9}

    assert torch.all(
        _process_per_edge_type_cutoff(type_names, per_edge_type_cutoff, r_max=4.0)
        == torch.as_tensor(
            [[2.0, 2.0, 2.0], [4.0, 3.5, 3.7], [3.9, 3.9, 3.9]], dtype=_GLOBAL_DTYPE
        )
    )

    # Ignores unspecified atoms
    type_names = ["H", "C", "O"]
    per_edge_type_cutoff = {
        "H": 2.0,
        "C": {"H": 4.0, "C": 3.5, "O": 3.7, "N": 3.7},
        "O": 3.9,
    }

    assert torch.all(
        _process_per_edge_type_cutoff(type_names, per_edge_type_cutoff, r_max=4.0)
        == torch.as_tensor(
            [[2.0, 2.0, 2.0], [4.0, 3.5, 3.7], [3.9, 3.9, 3.9]], dtype=_GLOBAL_DTYPE
        )
    )


def test_non_periodic_edge(CH3CHO):
    atoms, data = CH3CHO
    # check edges
    for edge in range(AtomicDataDict.num_edges(data)):
        real_displacement = (
            atoms.positions[data["edge_index"][1, edge]]
            - atoms.positions[data["edge_index"][0, edge]]
        )
        assert torch.allclose(
            with_edge_vectors_(data)["edge_vectors"][edge],
            torch.as_tensor(real_displacement, dtype=torch.get_default_dtype()),
        )


def test_periodic_edge():
    atoms = ase.build.bulk("Cu", "fcc")
    dist = np.linalg.norm(atoms.cell[0])
    data = compute_neighborlist_(from_ase(atoms), r_max=1.05 * dist)
    edge_vecs = with_edge_vectors_(data)["edge_vectors"]
    assert edge_vecs.shape == (12, 3)  # 12 neighbors in close-packed bulk
    assert torch.allclose(
        edge_vecs.norm(dim=-1), torch.as_tensor(dist, dtype=torch.get_default_dtype())
    )


@pytest.mark.parametrize("periodic", [True, False])
def test_positions_grad(periodic, CH3CHO, Cu_bulk):

    if periodic:
        atoms, data = Cu_bulk
    else:
        atoms, data = CH3CHO

    data["pos"].requires_grad_(True)
    data = with_edge_vectors_(data)
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
        data = with_edge_vectors_(data)
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
    data = compute_neighborlist_(
        from_ase(atoms), r_max=2.9
    )  # first shell dist is 2.864 A
    # Check number of neighbors:
    _, neighbor_count = np.unique(data["edge_index"][0].numpy(), return_counts=True)
    assert (neighbor_count == 6).all()  # 6 neighbors
    # Check not periodic in z
    assert torch.allclose(
        with_edge_vectors_(data)["edge_vectors"][:, 2],
        torch.zeros(
            AtomicDataDict.num_edges(data),
            dtype=with_edge_vectors_(data)["edge_vectors"].dtype,
        ),
    )


def test_basic(model_dtype):
    type_names = ["A", "B", "C", "D"]
    with torch_default_dtype(dtype_from_name(model_dtype)):

        node_type = NodeTypeEmbed(type_names=type_names, num_features=13)
        save = SaveForOutput(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field="saved",
            irreps_in=node_type.irreps_out,
        )
        linear = AtomwiseLinear(irreps_in=save.irreps_out)
        model = GraphModel(
            SequentialGraphNetwork(
                {
                    "one_hot": node_type,
                    "save": save,
                    "linear": linear,
                }
            ),
        )
    out = model(
        {
            AtomicDataDict.POSITIONS_KEY: torch.randn(5, 3),
            AtomicDataDict.EDGE_INDEX_KEY: torch.LongTensor([[0, 1], [1, 0]]),
            AtomicDataDict.ATOM_TYPE_KEY: torch.LongTensor([0, 0, 1, 2, 0]),
        }
    )
    saved = out["saved"]
    assert saved.shape == (5, 13)
