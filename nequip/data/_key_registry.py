"""_key_registry.py: manage information about what kinds of data different keys refer to."""

from typing import Dict, Set, Sequence

import numpy as np
import torch

from e3nn.io import CartesianTensor

from . import _keys
from . import AtomicDataDict


# === Key Registration ===

_DEFAULT_LONG_FIELDS: Set[str] = {
    _keys.EDGE_INDEX_KEY,
    _keys.EDGE_TYPE_KEY,
    _keys.ATOMIC_NUMBERS_KEY,
    _keys.ATOM_TYPE_KEY,
    _keys.BATCH_KEY,
    _keys.NUM_NODES_KEY,
}
_DEFAULT_GRAPH_FIELDS: Set[str] = {
    _keys.TOTAL_ENERGY_KEY,
    _keys.FREE_ENERGY_KEY,
    _keys.STRESS_KEY,
    _keys.VIRIAL_KEY,
    _keys.PBC_KEY,
    _keys.CELL_KEY,
    _keys.NUM_NODES_KEY,
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
    _keys.NORM_LENGTH_KEY,
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
    r"""Register fields as being per-frame, per-atom, per-edge, long dtype and/or Cartesian tensors."""
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
        cartesian_tensor_fields={
            prefix + e: v for e, v in _CARTESIAN_TENSOR_FIELDS.items()
        },
    )


def get_field_type(field: str) -> str:
    if field in _GRAPH_FIELDS:
        return "graph"
    elif field in _NODE_FIELDS:
        return "node"
    elif field in _EDGE_FIELDS:
        return "edge"
    else:
        raise KeyError(f"Unregistered field {field} found")


# === abbreviations ===
ABBREV = {
    _keys.TOTAL_ENERGY_KEY: "E",
    _keys.PER_ATOM_ENERGY_KEY: "Ei",
    _keys.FORCE_KEY: "F",
    _keys.TOTAL_MAGMOM_KEY: "M",
    _keys.CHARGE_KEY: "Q",
    _keys.POLARIZATION_KEY: "pol",
    _keys.BORN_CHARGE_KEY: "Z*",
    _keys.DIELECTRIC_KEY: "ε",
}


#  === AtomicData ===


def _process_dict(data):
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
        pbc = False

    if isinstance(pbc, bool):
        pbc = (pbc,) * 3
    elif isinstance(pbc, torch.Tensor):
        assert len(pbc) == 3 or pbc.shape[1] == 3  # account for batch dims
    else:
        assert len(pbc) == 3, pbc

    if cell is not None:
        # the reshape accounts for both (3, 3) or (N_frames, 3, 3) shaped cells
        data[_keys.CELL_KEY] = torch.as_tensor(
            cell, dtype=torch.get_default_dtype()
        ).reshape(-1, 3, 3)

    if pbc is not None:
        data[_keys.PBC_KEY] = torch.as_tensor(pbc, dtype=torch.bool).reshape(-1, 3)

    # == Deal with _some_ dtype issues ==
    for k, v in data.items():
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
            # Guerantee all values are torch.Tensors
            raise TypeError(
                f"Value for field {k} was of unsupported type {type(v)} (value was {v})"
            )

        # make sure evrything is contiguous
        data[k] = data[k].contiguous()

    # == get useful data properties ==
    if _keys.NUM_NODES_KEY in data:
        N_frames = AtomicDataDict.num_frames(data)
    else:
        N_frames = 1
    N_nodes = AtomicDataDict.num_nodes(data)
    if _keys.EDGE_INDEX_KEY in data:
        N_edges = AtomicDataDict.num_edges(data)
    else:
        N_edges = None

    # == Cartesian tensor field reshapes (ensure batch dimension present) ==

    # IMPORTANT: the following reshape logic only applies to rank-2 Cartesian tensor fields
    for k, v in data.items():
        if k in _CARTESIAN_TENSOR_FIELDS:
            # enforce (N_frames, 3, 3) shape for graph fields, e.g. stress, virial
            # remembering to handle ASE-style 6 element Voigt order stress
            if k in _GRAPH_FIELDS:
                err_msg = f"bad shape {v.shape} for {k} registered as a Cartesian tensor graph field---please note that only rank-2 Cartesian tensors are currently supported"
                if v.dim() == 1:  # two possibilities
                    if v.shape == (6,):
                        assert k in (_keys.STRESS_KEY, _keys.VIRIAL_KEY)
                        data[k] = _voigt_6_to_full_3x3_stress(v).reshape(1, 3, 3)
                    elif v.shape == (9,):
                        data[k] = v.reshape(1, 3, 3)
                    else:
                        raise RuntimeError(err_msg)
                elif v.dim() == 2:  # three cases
                    if v.shape == (N_frames, 6):
                        raise NotImplementedError(
                            f"File a GitHub issue if the parsing of shape signature (N_frames, 6) is required for {k}"
                        )
                    elif v.shape == (N_frames, 9):
                        data[k] = v.reshape((N_frames, 3, 3))
                    elif v.shape == (3, 3):
                        data[k] = v.reshape((1, 3, 3))
                    else:
                        raise RuntimeError(err_msg)
                elif v.dim() == 3:  # one possibility - it's already correctly shaped
                    assert v.shape == (N_frames, 3, 3), err_msg
            # enforce (N_nodes, 3, 3) shape for node fields, e.g. Born effective charges
            elif k in _NODE_FIELDS:
                err_msg = f"bad shape {v.shape} for {k} registered as a Cartesian tensor node field---please note that only rank-2 Cartesian tensors are currently supported"
                if v.dim() == 1:  # one possibility
                    assert v.shape[0] == 9, err_msg
                    data[k] = v.reshape((1, 3, 3))
                elif v.dim() == 2:  # three possibilities
                    if v.shape == (3, 3):
                        data[k] = v.reshape(-1, 3, 3)
                    elif v.shape == (N_nodes, 9):
                        data[k] = v.reshape(N_nodes, 3, 3)
                    elif v.shape == (N_nodes, 6):  # i.e. Voigt format
                        # TODO (maybe): this is inefficient, but who is going to train on per-atom stresses except for toy training runs?
                        data[k] = torch.stack(
                            [_voigt_6_to_full_3x3_stress(vec6) for vec6 in v]
                        )
                    else:
                        raise RuntimeError(err_msg)
                elif v.dim() == 3:  # one possibility
                    assert v.shape == (N_nodes, 3, 3), err_msg
                else:
                    raise RuntimeError(err_msg)
            else:
                raise RuntimeError(
                    f"{k} registered as a Cartesian tensor field was not registered as either a graph or node field"
                )

    # == general shape checks ==
    for k, v in data.items():
        if len(v.shape) == 0:
            data[k] = v.unsqueeze(-1)
            v = data[k]

        if k in _GRAPH_FIELDS:
            assert (
                v.shape[0] == N_frames
            ), f"Leading dimension of registered graph field {k} should be {N_frames}, but found shape {v.shape}."

            # TODO: consider removing -- why not?
            if v.dim() == 1 and k not in [_keys.NUM_NODES_KEY]:
                data[k] = v.reshape((N_frames, 1))

        elif k in _NODE_FIELDS:
            assert (
                v.shape[0] == N_nodes
            ), f"Leading dimension of registered node field {k} should be {N_nodes}, but found shape {v.shape}."

            # TODO: consider removing -- why not?
            if v.dim() == 1 and k not in [
                _keys.BATCH_KEY,
                _keys.ATOMIC_NUMBERS_KEY,
                _keys.ATOM_TYPE_KEY,
            ]:
                data[k] = v.reshape((N_nodes, 1))

        elif k in _EDGE_FIELDS:
            if N_edges is None:
                raise ValueError(
                    f"Inconsistent data -- {k} was registered as an edge field, but no edge indices found."
                )
            else:
                assert (
                    v.shape[0] == N_edges
                ), f"Leading dimension of registered edge field {k} should be {N_edges}, but found shape {v.shape}."

    # == specific checks for basic properties (pos, cell) ==
    pos = data[_keys.POSITIONS_KEY]
    assert pos.dim() == 2 and pos.shape[1] == 3

    if _keys.CELL_KEY in data:
        cell = data[_keys.CELL_KEY]
        assert cell.dim() == 3 and cell.shape == (N_frames, 3, 3)
        assert cell.dtype == pos.dtype
        pbc = data[_keys.PBC_KEY]
        assert pbc.dim() == 2 and pbc.shape == (N_frames, 3)

    return data


def _voigt_6_to_full_3x3_stress(voigt_stress):
    """
    Form a 3x3 stress matrix from a 6 component vector in Voigt notation
    """
    return torch.Tensor(
        [
            [voigt_stress[0], voigt_stress[5], voigt_stress[4]],
            [voigt_stress[5], voigt_stress[1], voigt_stress[3]],
            [voigt_stress[4], voigt_stress[3], voigt_stress[2]],
        ]
    )
