"""nequip.data.jit: TorchScript functions for dealing with AtomicData.

These TorchScript functions operate on ``Dict[str, torch.Tensor]`` representations
of the ``AtomicData`` class which are produced by ``AtomicData.to_AtomicDataDict()``.

Authors: Albert Musaelian
"""
from typing import Dict, Any

import torch
import torch.jit

from e3nn import o3

# Make the keys available in this module
from ._keys import *  # noqa: F403, F401

# Also import the module to use in TorchScript, this is a hack to avoid bug:
# https://github.com/pytorch/pytorch/issues/52312
from . import _keys

# Define a type alias
Type = Dict[str, torch.Tensor]


def validate_keys(keys, graph_required=True):
    # Validate combinations
    if graph_required:
        if not (_keys.POSITIONS_KEY in keys and _keys.EDGE_INDEX_KEY in keys):
            raise KeyError("At least pos and edge_index must be supplied")
    if _keys.EDGE_CELL_SHIFT_KEY in keys and "cell" not in keys:
        raise ValueError("If `edge_cell_shift` given, `cell` must be given.")
    if _keys.ATOMIC_NUMBERS_KEY in keys and _keys.SPECIES_INDEX_KEY in keys:
        raise ValueError(
            "'atomic_numbers' and 'species_index' cannot be simultaneously provided"
        )


_SPECIAL_IRREPS = [None]


def _fix_irreps_dict(d: Dict[str, Any]):
    return {k: (i if i in _SPECIAL_IRREPS else o3.Irreps(i)) for k, i in d.items()}


def _irreps_compatible(ir1: Dict[str, o3.Irreps], ir2: Dict[str, o3.Irreps]):
    return all(ir1[k] == ir2[k] for k in ir1 if k in ir2)


@torch.jit.script
def with_edge_vectors(data: Type) -> Type:
    """Compute the edge displacement vectors for a graph.

    If ``data.pos.requires_grad`` and/or ``data.cell.requires_grad``, this
    method will return edge vectors correctly connected in the autograd graph.

    Returns:
        Tensor [n_edges, 3] edge displacement vectors
    """
    if _keys.EDGE_VECTORS_KEY in data:
        return data
    else:
        # Build it dynamically
        # Note that this is
        # (1) backwardable, because everything (pos, cell, shifts)
        #     is Tensors.
        # (2) works on a Batch constructed from AtomicData
        pos = data[_keys.POSITIONS_KEY]
        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
        has_cell: bool = (_keys.CELL_KEY in data) and (
            _keys.EDGE_CELL_SHIFT_KEY in data
        )
        has_batch: bool = _keys.BATCH_KEY in data
        if has_cell:
            cell = data[_keys.CELL_KEY]
            edge_cell_shift = data[_keys.EDGE_CELL_SHIFT_KEY]
            if cell.shape[0] > 1 and has_batch:
                batch = data[_keys.BATCH_KEY]
                # Cell has a batch dimension
                edge_vec = edge_vec + torch.einsum(
                    "ni,nij->nj", edge_cell_shift, cell[batch[edge_index[0]]]
                )
                # TODO: is there a more efficient way to do the above without
                # creating an [n_edge] and [n_edge, 3, 3] tensor?
            else:
                # Cell has either no batch dimension, or a uselesss one,
                # so we can avoid creating the large intermediate cell tensor.
                edge_vec = edge_vec + torch.einsum(
                    "ni,ij->nj",
                    edge_cell_shift,
                    cell.squeeze(0),  # remove batch dimension
                )
        data[_keys.EDGE_VECTORS_KEY] = edge_vec
        return data


@torch.jit.script
def with_batch(data: Type) -> Type:
    """Get batch Tensor.

    If this AtomicDataPrimitive has no ``batch``, one of all zeros will be
    allocated and returned.
    """
    if _keys.BATCH_KEY in data:
        return data
    else:
        pos = data[_keys.POSITIONS_KEY]
        batch = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
        data[_keys.BATCH_KEY] = batch
        return data
