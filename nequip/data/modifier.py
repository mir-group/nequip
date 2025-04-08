# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""
Data statistics and metrics managers work with BaseModifier, its subclasses (and perhaps classes that mimic their behavior) under the hood.
The action of a modifier:
 - `AtomicDataDict` -> `torch.Tensor`  (for data statistics), or
 - `AtomicDataDict`, `AtomicDataDict` -> `torch.Tensor`, `torch.Tensor`  (for metrics)
It should implement
  - `__str__()` for automatic naming,
  - `type()` property for data processing logic,
  - `__call__()` for its action, or
  - optionally `_func()` if the same action is to be applied for both data dicts (when used in metrics)
"""

import torch
from . import AtomicDataDict, _key_registry
from nequip.nn.utils import with_edge_vectors_
from typing import Optional, Union, List


class BaseModifier:

    def __init__(self, field: str) -> None:
        self.field = field

    def _func(self, data: AtomicDataDict.Type) -> torch.Tensor:
        return data[self.field]

    def __call__(
        self, data1: AtomicDataDict.Type, data2: Optional[AtomicDataDict.Type] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if data2 is None:
            return self._func(data1)
        else:
            return self._func(data1), self._func(data2)

    def __str__(self) -> str:
        return _key_registry.ABBREV.get(self.field, self.field)

    @property
    def type(self) -> str:
        return _key_registry.get_field_type(self.field)


class PerAtomModifier(BaseModifier):
    """Normalizes a graph field by the number of atoms (nodes) in the graph.

    Args:
        field (str): graph field to be normalized (e.g. ``total_energy``)
        factor (float): optional factor to scale the field by (e.g. for unit conversions, etc)
    """

    def __init__(self, field: str, factor: Optional[float] = None) -> None:
        assert field in _key_registry._GRAPH_FIELDS
        super().__init__(field)
        self._factor = factor

    def _func(self, data: AtomicDataDict.Type) -> torch.Tensor:
        num_atoms = (
            data[AtomicDataDict.NUM_NODES_KEY].reciprocal().reshape(-1)
        )  # (N_graph,)
        normed = torch.einsum("n..., n -> n...", data[self.field], num_atoms)
        if self._factor is not None:
            normed = self._factor * normed
        return normed

    def __str__(self) -> str:
        return "per_atom_" + _key_registry.ABBREV.get(self.field, self.field)


class EdgeLengths(BaseModifier):
    """Get edge lengths from an ``AtomicDataDict``."""

    def __init__(self) -> None:
        super().__init__(AtomicDataDict.EDGE_INDEX_KEY)

    def _func(self, data: AtomicDataDict.Type) -> torch.Tensor:
        data = with_edge_vectors_(data, with_lengths=True)
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

    def _func(self, data: AtomicDataDict.Type) -> torch.Tensor:
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
