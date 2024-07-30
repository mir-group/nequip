"""_key_registry.py: manage information about what kinds of data different keys refer to.
"""

import warnings
from typing import Any, Dict, Set, Sequence

import numpy as np
import torch

from e3nn import o3
from e3nn.io import CartesianTensor

from . import _keys
from ._util import _TORCH_INTEGER_DTYPES

# == Irrep checking ==
# TODO: unclear if these three functions used, delete?
# I think `_process_dict` (which could have a better name)
# does what this was supposed to.


def validate_keys(keys, graph_required=True):
    # Validate combinations
    if graph_required:
        if not (_keys.POSITIONS_KEY in keys and _keys.EDGE_INDEX_KEY in keys):
            raise KeyError("At least pos and edge_index must be supplied")
    if _keys.EDGE_CELL_SHIFT_KEY in keys and "cell" not in keys:
        raise ValueError("If `edge_cell_shift` given, `cell` must be given.")


_SPECIAL_IRREPS = [None]


def _fix_irreps_dict(d: Dict[str, Any]):
    return {k: (i if i in _SPECIAL_IRREPS else o3.Irreps(i)) for k, i in d.items()}


def _irreps_compatible(ir1: Dict[str, o3.Irreps], ir2: Dict[str, o3.Irreps]):
    return all(ir1[k] == ir2[k] for k in ir1 if k in ir2)


# === Key Registration ===

_DEFAULT_LONG_FIELDS: Set[str] = {
    _keys.EDGE_INDEX_KEY,
    _keys.ATOMIC_NUMBERS_KEY,
    _keys.ATOM_TYPE_KEY,
    _keys.BATCH_KEY,
    _keys.BATCH_PTR_KEY,
}
_DEFAULT_GRAPH_FIELDS: Set[str] = {
    _keys.TOTAL_ENERGY_KEY,
    _keys.FREE_ENERGY_KEY,
    _keys.STRESS_KEY,
    _keys.VIRIAL_KEY,
    _keys.PBC_KEY,
    _keys.CELL_KEY,
    _keys.BATCH_PTR_KEY,
    _keys.TOTAL_MAGMOM_KEY,
    _keys.POLARIZATION_KEY,
    _keys.DIELECTRIC_KEY,
}
_DEFAULT_NODE_FIELDS: Set[str] = {
    _keys.POSITIONS_KEY,
    _keys.NODE_FEATURES_KEY,
    _keys.NODE_ATTRS_KEY,
    _keys.ATOMIC_NUMBERS_KEY,
    _keys.ATOM_TYPE_KEY,
    _keys.PER_ATOM_ENERGY_KEY,
    _keys.CHARGE_KEY,
    _keys.FORCE_KEY,
    _keys.PER_ATOM_STRESS_KEY,
    _keys.MAGMOM_KEY,
    _keys.DIPOLE_KEY,
    _keys.BORN_CHARGE_KEY,
    _keys.BATCH_KEY,
}
_DEFAULT_EDGE_FIELDS: Set[str] = {
    _keys.EDGE_CELL_SHIFT_KEY,
    _keys.EDGE_VECTORS_KEY,
    _keys.EDGE_LENGTH_KEY,
    _keys.EDGE_VECTORS_F64_KEY,
    _keys.EDGE_LENGTH_F64_KEY,
    _keys.EDGE_ATTRS_KEY,
    _keys.EDGE_EMBEDDING_KEY,
    _keys.EDGE_FEATURES_KEY,
    _keys.EDGE_CUTOFF_KEY,
    _keys.EDGE_ENERGY_KEY,
}
_DEFAULT_CARTESIAN_TENSOR_FIELDS: Dict[str, str] = {
    _keys.STRESS_KEY: "ij=ji",
    _keys.VIRIAL_KEY: "ij=ji",
    _keys.PER_ATOM_STRESS_KEY: "ij=ji",
    _keys.DIELECTRIC_KEY: "ij=ji",
    _keys.BORN_CHARGE_KEY: "ij",
}
_LONG_FIELDS: Set[str] = set(_DEFAULT_LONG_FIELDS)
_GRAPH_FIELDS: Set[str] = set(_DEFAULT_GRAPH_FIELDS)
_NODE_FIELDS: Set[str] = set(_DEFAULT_NODE_FIELDS)
_EDGE_FIELDS: Set[str] = set(_DEFAULT_EDGE_FIELDS)
_CARTESIAN_TENSOR_FIELDS: Dict[str, str] = dict(_DEFAULT_CARTESIAN_TENSOR_FIELDS)


def register_fields(
    node_fields: Sequence[str] = [],
    edge_fields: Sequence[str] = [],
    graph_fields: Sequence[str] = [],
    long_fields: Sequence[str] = [],
    cartesian_tensor_fields: Dict[str, str] = {},
) -> None:
    r"""Register fields as being per-atom, per-edge, or per-frame.

    Args:
        node_permute_fields: fields that are equivariant to node permutations.
        edge_permute_fields: fields that are equivariant to edge permutations.
    """
    node_fields: set = set(node_fields)
    edge_fields: set = set(edge_fields)
    graph_fields: set = set(graph_fields)
    long_fields: set = set(long_fields)

    # error checking: prevents registering fields as contradictory types
    # potentially unregistered fields
    assert len(node_fields.intersection(edge_fields)) == 0
    assert len(node_fields.intersection(graph_fields)) == 0
    assert len(edge_fields.intersection(graph_fields)) == 0
    # already registered fields
    assert len(_NODE_FIELDS.intersection(edge_fields)) == 0
    assert len(_NODE_FIELDS.intersection(graph_fields)) == 0
    assert len(_EDGE_FIELDS.intersection(node_fields)) == 0
    assert len(_EDGE_FIELDS.intersection(graph_fields)) == 0
    assert len(_GRAPH_FIELDS.intersection(edge_fields)) == 0
    assert len(_GRAPH_FIELDS.intersection(node_fields)) == 0

    # check that Cartesian tensor fields to add are rank-2 (higher ranks not supported)
    for cart_tensor_key in cartesian_tensor_fields:
        cart_tensor_rank = len(
            CartesianTensor(cartesian_tensor_fields[cart_tensor_key]).indices
        )
        if cart_tensor_rank != 2:
            raise NotImplementedError(
                f"Only rank-2 tensor data processing supported, but got {cart_tensor_key} is rank {cart_tensor_rank}. Consider raising a GitHub issue if higher-rank tensor data processing is desired."
            )

    # update fields
    _NODE_FIELDS.update(node_fields)
    _EDGE_FIELDS.update(edge_fields)
    _GRAPH_FIELDS.update(graph_fields)
    _LONG_FIELDS.update(long_fields)
    _CARTESIAN_TENSOR_FIELDS.update(cartesian_tensor_fields)


def deregister_fields(*fields: Sequence[str]) -> None:
    r"""Deregister a field registered with ``register_fields``.

    Silently ignores fields that were never registered to begin with.

    Args:
        *fields: fields to deregister.
    """
    for f in fields:
        assert f not in _DEFAULT_NODE_FIELDS, "Cannot deregister built-in field"
        assert f not in _DEFAULT_EDGE_FIELDS, "Cannot deregister built-in field"
        assert f not in _DEFAULT_GRAPH_FIELDS, "Cannot deregister built-in field"
        assert f not in _DEFAULT_LONG_FIELDS, "Cannot deregister built-in field"
        assert (
            f not in _DEFAULT_CARTESIAN_TENSOR_FIELDS
        ), "Cannot deregister built-in field"

        _NODE_FIELDS.discard(f)
        _EDGE_FIELDS.discard(f)
        _GRAPH_FIELDS.discard(f)
        _LONG_FIELDS.discard(f)
        _CARTESIAN_TENSOR_FIELDS.pop(f, None)


def _register_field_prefix(prefix: str) -> None:
    """Re-register all registered fields as the same type, but with `prefix` added on."""
    assert prefix.endswith("_")
    register_fields(
        node_fields=[prefix + e for e in _NODE_FIELDS],
        edge_fields=[prefix + e for e in _EDGE_FIELDS],
        graph_fields=[prefix + e for e in _GRAPH_FIELDS],
        long_fields=[prefix + e for e in _LONG_FIELDS],
    )


#  === AtomicData ===


def _process_dict(data, ignore_fields=[]):
    """Convert a dict of data into correct dtypes/shapes according to key"""
    data = data.copy()

    # == Deal with basic variables pos, cell, pbc ==
    assert _keys.POSITIONS_KEY in data, "At least pos must be supplied"

    cell = data.get(_keys.CELL_KEY, None)
    pbc = data.get(_keys.PBC_KEY, None)
    if pbc is None:
        if cell is not None:
            raise ValueError(
                "A cell was provided, but pbc's were not. Please explicitly provide PBC."
            )
        # there are no PBC if cell and pbc are not provided
        pbc = False

    if isinstance(pbc, bool):
        pbc = (pbc,) * 3
    else:
        assert len(pbc) == 3

    if cell is not None:
        data[_keys.CELL_KEY] = torch.as_tensor(
            cell, dtype=torch.get_default_dtype()
        ).view(-1, 3, 3)

    if pbc is not None:
        data[_keys.PBC_KEY] = torch.as_tensor(pbc, dtype=torch.bool).view(-1, 3)

    # == Deal with _some_ dtype issues ==
    for k, v in data.items():
        if k in ignore_fields:
            continue

        if k in _LONG_FIELDS:
            # Any property used as an index must be long (or byte or bool, but those are not relevant for atomic scale systems)
            # int32 would pass later checks, but is actually disallowed by torch
            data[k] = torch.as_tensor(v, dtype=torch.long)
        elif isinstance(v, bool):
            data[k] = torch.as_tensor(v)
        elif isinstance(v, np.ndarray):
            if np.issubdtype(v.dtype, np.floating):
                data[k] = torch.as_tensor(v, dtype=torch.get_default_dtype())
            else:
                data[k] = torch.as_tensor(v)
        elif isinstance(v, list):
            ele_dtype = np.array(v).dtype
            if np.issubdtype(ele_dtype, np.floating):
                data[k] = torch.as_tensor(v, dtype=torch.get_default_dtype())
            else:
                data[k] = torch.as_tensor(v)
        elif np.issubdtype(type(v), np.floating):
            # Force scalars to be tensors with a data dimension
            # This makes them play well with irreps
            data[k] = torch.as_tensor(v, dtype=torch.get_default_dtype())
        elif isinstance(v, torch.Tensor) and len(v.shape) == 0:
            # ^ this tensor is a scalar; we need to give it
            # a data dimension to play nice with irreps
            data[k] = v
        elif isinstance(v, torch.Tensor):
            # This is a tensor, so we just don't do anything except avoid the warning in the `else`
            pass
        else:
            warnings.warn(
                f"Value for field {k} was of unsupported type {type(v)} (value was {v})"
            )

    if _keys.BATCH_PTR_KEY in data:
        num_frames = len(data[_keys.BATCH_PTR_KEY]) - 1
    else:
        num_frames = 1

    for k, v in data.items():
        if k in ignore_fields:
            continue

        if len(v.shape) == 0:
            data[k] = v.unsqueeze(-1)
            v = data[k]

        if k in set.union(_NODE_FIELDS, _EDGE_FIELDS) and len(v.shape) == 1:
            data[k] = v.unsqueeze(-1)
            v = data[k]

        if (
            k in _NODE_FIELDS
            and _keys.POSITIONS_KEY in data
            and v.shape[0] != data[_keys.POSITIONS_KEY].shape[0]
        ):
            raise ValueError(
                f"{k} is a node field but has the wrong dimension {v.shape}"
            )
        elif (
            k in _EDGE_FIELDS
            and _keys.EDGE_INDEX_KEY in data
            and v.shape[0] != data[_keys.EDGE_INDEX_KEY].shape[1]
        ):
            raise ValueError(
                f"{k} is a edge field but has the wrong dimension {v.shape}"
            )
        elif k in _GRAPH_FIELDS:
            if num_frames > 1 and v.shape[0] != num_frames:
                raise ValueError(f"Wrong shape for graph property {k}")

    # validate shapes and dtypes
    assert (
        data[_keys.POSITIONS_KEY].dim() == 2 and data[_keys.POSITIONS_KEY].shape[1] == 3
    )

    if _keys.CELL_KEY in data and data[_keys.CELL_KEY] is not None:
        assert (data[_keys.CELL_KEY].shape == (3, 3)) or (
            data[_keys.CELL_KEY].dim() == 3 and data[_keys.CELL_KEY].shape[1:] == (3, 3)
        )
        assert data[_keys.CELL_KEY].dtype == data[_keys.POSITIONS_KEY].dtype

    if _keys.ATOMIC_NUMBERS_KEY in data and data[_keys.ATOMIC_NUMBERS_KEY] is not None:
        assert data[_keys.ATOMIC_NUMBERS_KEY].dtype in _TORCH_INTEGER_DTYPES
    if _keys.BATCH_KEY in data and data[_keys.BATCH_KEY] is not None:
        assert data[_keys.BATCH_KEY].dim() == 1 and data[_keys.BATCH_KEY].shape[
            0
        ] == len(data[_keys.POSITIONS_KEY])
        # Check that there are the right number of cells
        if _keys.CELL_KEY in data and data[_keys.CELL_KEY] is not None:
            cell = data[_keys.CELL_KEY].view(-1, 3, 3)
            assert cell.shape[0] == num_frames
    return data
