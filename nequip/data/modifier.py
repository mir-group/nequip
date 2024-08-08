"""
Data statistics and metrics managers work with BaseModifier, its subclasses (and perhaps classes that mimic their behavior) under the hood.
The action of a modifier: `AtomicDataDict` -> torch.Tensor

It should implement
  - __str__() for automatic naming, and
  - type() property for data processing logic
"""

import torch
from . import AtomicDataDict, _key_registry


class BaseModifier:

    def __init__(self, field: str) -> None:
        self.field = field

    def __call__(self, data: AtomicDataDict.Type) -> torch.Tensor:
        return data[self.field]

    def __str__(self) -> str:
        return _key_registry.ABBREV.get(self.field, self.field)

    @property
    def type(self) -> str:
        return _key_registry.get_field_type(self.field)


class PerAtomModifier(BaseModifier):
    """Normalizes a graph field by the number of atoms (nodes) in the graph.

    Args:
        field (str): graph field to be normalized (e.g. ``total_energy``)
    """

    def __init__(self, field: str) -> None:
        assert field in _key_registry._GRAPH_FIELDS
        super().__init__(field)

    def __call__(self, data: AtomicDataDict.Type) -> torch.Tensor:
        num_atoms = (
            data[AtomicDataDict.NUM_NODES_KEY].reciprocal().reshape(-1)
        )  # (N_graph,)
        return torch.einsum("n..., n -> n...", data[self.field], num_atoms)

    def __str__(self) -> str:
        return "per_atom_" + _key_registry.ABBREV.get(self.field, self.field)


class EdgeLengths(BaseModifier):
    """Get edge lengths from an ``AtomicDataDict``."""

    def __init__(self) -> None:
        super().__init__(AtomicDataDict.EDGE_INDEX_KEY)

    def __call__(self, data: AtomicDataDict.Type) -> torch.Tensor:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        return data[AtomicDataDict.EDGE_LENGTH_KEY]

    def __str__(self) -> str:
        return "edge_lengths"

    @property
    def type(self) -> str:
        return "edge"


class NumNeighbors(BaseModifier):
    """Get number of neighbors from an ``AtomicDataDict``."""

    def __init__(self) -> None:
        super().__init__(AtomicDataDict.EDGE_INDEX_KEY)

    def __call__(self, data: AtomicDataDict.Type) -> torch.Tensor:
        counts = torch.unique(
            data[AtomicDataDict.EDGE_INDEX_KEY][0],
            sorted=True,
            return_counts=True,
        )[1]
        # in case the cutoff is small and some nodes have no neighbors,
        # we need to pad `counts` up to the right length
        counts = torch.nn.functional.pad(
            counts, pad=(0, len(data[AtomicDataDict.POSITIONS_KEY]) - len(counts))
        )
        return counts

    def __str__(self) -> str:
        return "num_neighbors"

    @property
    def type(self) -> str:
        return "node"
