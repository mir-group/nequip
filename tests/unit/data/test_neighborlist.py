import numpy as np
import pytest
import torch
from ase import Atoms
import ase.build
from ase.calculators.singlepoint import SinglePointCalculator

from nequip.data import AtomicDataDict, compute_neighborlist_, from_ase, from_dict
from nequip.nn.utils import with_edge_vectors_
from nequip.data._nl import (
    NEIGHBORLIST_BACKEND_ALCHEMIOPS,
    NEIGHBORLIST_BACKEND_ASE,
    NEIGHBORLIST_BACKEND_MATSCIPY,
    NEIGHBORLIST_BACKEND_VESIN,
)

try:
    import vesin  # noqa: F401

    VESIN_AVAILABLE = True
except ImportError:
    VESIN_AVAILABLE = False

try:
    import nvalchemiops  # noqa: F401

    ALCHEMIOPS_AVAILABLE = True
except ImportError:
    ALCHEMIOPS_AVAILABLE = False


BACKEND_CASES = [NEIGHBORLIST_BACKEND_ASE, NEIGHBORLIST_BACKEND_MATSCIPY]
if VESIN_AVAILABLE:
    BACKEND_CASES.append(NEIGHBORLIST_BACKEND_VESIN)
if ALCHEMIOPS_AVAILABLE:
    BACKEND_CASES.append(NEIGHBORLIST_BACKEND_ALCHEMIOPS)


@pytest.fixture(scope="function")
def CuFcc():
    atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = SinglePointCalculator(
        atoms, **{"forces": np.random.random((len(atoms), 3))}
    )
    data = compute_neighborlist_(
        from_ase(atoms),
        r_max=4.0,
        backend=NEIGHBORLIST_BACKEND_ASE,
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
        pos=coords,
        cell=lattice,
        pbc=True,
    )
    data = compute_neighborlist_(
        from_dict(points),
        r_max=r_max,
        backend=NEIGHBORLIST_BACKEND_ASE,
    )
    return r_max, points, data


def edge_index_set_equiv(a, b):
    # [[0, 1], [1, 0]] -> {(0, 1), (1, 0)}
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    return set(zip(a[0], a[1])) == set(zip(b[0], b[1]))


def clone_data_dict(data):
    return {k: v.clone() for k, v in data.items()}


def test_silicon_neighbors(Si):
    r_max, points, data = Si
    test_data = compute_neighborlist_(
        from_dict(points),
        r_max=r_max,
    )
    edge_index = test_data[AtomicDataDict.EDGE_INDEX_KEY]
    edge_index_true = torch.LongTensor(
        [[0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]]
    )
    assert edge_index_set_equiv(edge_index, edge_index_true)
    assert edge_index_set_equiv(data[AtomicDataDict.EDGE_INDEX_KEY], edge_index_true)


@pytest.mark.parametrize("nl_method", BACKEND_CASES)
def test_no_neighbors(nl_method):
    # isolated atom
    H = Atoms("H", positions=[[0, 0, 0]], cell=20 * np.eye(3))
    data = compute_neighborlist_(from_ase(H), r_max=2.5, backend=nl_method)
    assert data[AtomicDataDict.EDGE_INDEX_KEY].numel() == 0
    assert data[AtomicDataDict.EDGE_CELL_SHIFT_KEY].numel() == 0

    # cutoff smaller than interatomic distance
    Cu = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
    data = compute_neighborlist_(from_ase(Cu), r_max=2.5, backend=nl_method)
    assert data[AtomicDataDict.EDGE_INDEX_KEY].numel() == 0
    assert data[AtomicDataDict.EDGE_CELL_SHIFT_KEY].numel() == 0


@pytest.mark.parametrize("backend", BACKEND_CASES)
@pytest.mark.parametrize(
    "input_batching", ["unbatched", "batched_single", "batched_multi"]
)
@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_neighborlist_contracts_and_consistency(
    backend, input_batching, device, CH3CHO, CuFcc
):
    """Test neighborlist contracts for batch-state, device, input-preservation, and ASE edge consistency."""
    ch3cho_atoms, _ = CH3CHO
    cufcc_atoms, _ = CuFcc
    r_max = 4.5

    for atoms in [ch3cho_atoms, cufcc_atoms]:
        if input_batching == "unbatched":
            data = from_ase(atoms.copy())
        elif input_batching == "batched_single":
            data = AtomicDataDict.with_batch_(from_ase(atoms.copy()))
        else:
            atoms_a = atoms.copy()
            atoms_b = atoms.copy()
            atoms_c = atoms.copy()
            atoms_b.rattle(stdev=0.01)
            atoms_c.rattle(stdev=0.01)
            data = AtomicDataDict.batched_from_list(
                [
                    AtomicDataDict.with_batch_(from_ase(atoms_a)),
                    AtomicDataDict.with_batch_(from_ase(atoms_b)),
                    AtomicDataDict.with_batch_(from_ase(atoms_c)),
                ]
            )
        data = AtomicDataDict.to_(data, device=device)

        backend_input = clone_data_dict(data)
        backend_input_clone = clone_data_dict(backend_input)
        ase_input = clone_data_dict(data)

        out = compute_neighborlist_(
            backend_input,
            r_max=r_max,
            backend=backend,
        )
        out_ase = compute_neighborlist_(
            ase_input,
            r_max=r_max,
            backend=NEIGHBORLIST_BACKEND_ASE,
        )
        # assess batch-state contract: output batching matches input batching mode
        assert AtomicDataDict.is_batched(out) is (input_batching != "unbatched")
        for key, value in backend_input_clone.items():
            # assess input-preservation contract: original keys are still present
            assert key in out
            # assess input-preservation contract: original tensor values are unchanged
            torch.testing.assert_close(out[key], value)
        # assess consistency contract: edge connectivity matches ASE for the same input
        assert edge_index_set_equiv(
            out[AtomicDataDict.EDGE_INDEX_KEY],
            out_ase[AtomicDataDict.EDGE_INDEX_KEY],
        )
        for v in out.values():
            # assess device contract: every output tensor stays on the input device
            assert v.device.type == device


@pytest.mark.parametrize("backend", BACKEND_CASES)
def test_neighborlist_wrapped_unwrapped_consistency(backend, CuFcc):
    atoms, _ = CuFcc
    atoms = atoms.copy()
    atoms.rattle(stdev=0.01, seed=0)
    r_max = 4.5

    atoms_wrapped = atoms.copy()
    atoms_wrapped.wrap()

    atoms_unwrapped = atoms_wrapped.copy()
    n_atoms = len(atoms_unwrapped)
    shift_coeffs = np.zeros((n_atoms, 3), dtype=np.int64)
    shift_coeffs[:, 0] = np.arange(n_atoms, dtype=np.int64) % 3 - 1
    shift_coeffs[:, 1] = np.arange(n_atoms, dtype=np.int64) % 2
    shift_coeffs[:, 2] = -(np.arange(n_atoms, dtype=np.int64) % 2)
    atoms_unwrapped.positions = (
        atoms_unwrapped.positions + shift_coeffs @ atoms_unwrapped.cell.array
    )

    wrapped_data = compute_neighborlist_(
        from_ase(atoms_wrapped), r_max=r_max, backend=backend
    )
    unwrapped_data = compute_neighborlist_(
        from_ase(atoms_unwrapped), r_max=r_max, backend=backend
    )

    wrapped_data = with_edge_vectors_(wrapped_data, with_lengths=True)
    unwrapped_data = with_edge_vectors_(unwrapped_data, with_lengths=True)

    wrapped_lengths = torch.sort(wrapped_data[AtomicDataDict.EDGE_LENGTH_KEY].view(-1))[
        0
    ]
    unwrapped_lengths = torch.sort(
        unwrapped_data[AtomicDataDict.EDGE_LENGTH_KEY].view(-1)
    )[0]

    # wrapped and unwrapped configurations should produce the same neighbor count
    assert wrapped_lengths.size(0) == unwrapped_lengths.size(0)
    # wrapped and unwrapped configurations should produce the same neighbor distances
    torch.testing.assert_close(
        wrapped_lengths, unwrapped_lengths, rtol=1e-14, atol=1e-14
    )
