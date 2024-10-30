"""_key_registry.py: manage information about what kinds of data different keys refer to."""

from typing import Dict, Set, Sequence

from e3nn.io import CartesianTensor

from . import _keys


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
    graph_fields: Sequence[str] = [],
    node_fields: Sequence[str] = [],
    edge_fields: Sequence[str] = [],
    long_fields: Sequence[str] = [],
    cartesian_tensor_fields: Dict[str, str] = {},
) -> None:
    """Register custom fields as being per-frame, per-atom, per-edge, long dtype and/or Cartesian tensors.

    Args:
        graph_fields (Sequence[str]): per-frame fields
        node_fields (Sequence[str]): per-atom fields
        edge_fields (Sequence[str]): per-edge fields
        long_fields (Sequence[str]): long ``dtype`` fields
        cartesian_tensor_fields (Dict[str, str]): Cartesian tensor fields (both the name, and the ``formula`` must be provided, e.g. ``"ij=ji"``, see `e3nn docs <https://docs.e3nn.org/en/stable/api/io/cartesian_tensor.html>`_)
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
        ``*fields`` (Sequence[str]): fields to deregister.
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
