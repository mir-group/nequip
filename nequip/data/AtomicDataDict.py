# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""AtomicDataDict

A "static class" (set of functions) operating on `Dict[str, torch.Tensor]`,
aliased here as `AtomicDataDict.Type`.  By avoiding a custom object wrapping
this simple type we avoid unnecessary abstraction and ensure compatibility
with a broad range of PyTorch features and libraries (including TorchScript)
that natively handle dictionaries of tensors.

Each function in this module is sort of like a method on the `AtomicDataDict`
class, if there was such a class---"self" is just passed explicitly:

    AtomicDataDict.some_method(my_data)

Some standard fields:

    pos (Tensor [n_nodes, 3]): Positions of the nodes.
    edge_index (LongTensor [2, n_edges]): ``edge_index[0]`` is the per-edge
        index of the source node and ``edge_index[1]`` is the target node.
    edge_cell_shift (Tensor [n_edges, 3], optional): which periodic image
        of the target point each edge goes to, relative to the source point.
    cell (Tensor [1, 3, 3], optional): the periodic cell for
        ``edge_cell_shift`` as the three triclinic cell vectors.
    node_features (Tensor [n_atom, ...]): the input features of the nodes, optional
    node_attrs (Tensor [n_atom, ...]): the attributes of the nodes, for instance the atom type, optional
    batch (Tensor [n_atom]): the graph to which the node belongs, optional
    atomic_numbers (Tensor [n_atom]): optional
    atom_type (Tensor [n_atom]): optional
"""

from typing import Dict, Union, Tuple, List, Optional, Any

import torch
from e3nn.o3._irreps import Irreps

# Make the keys available in this module
from ._keys import *  # noqa: F403, F401

# Also import the module to use in TorchScript, this is a hack to avoid bug:
# https://github.com/pytorch/pytorch/issues/52312
from . import _keys

from . import _key_registry

# Define a type alias
Type = Dict[str, torch.Tensor]
# A type representing ASE-style periodic boundary condtions, which can be partial (the tuple case)
PBCType = Union[bool, Tuple[bool, bool, bool]]

# == Irrep checking ==

_SPECIAL_IRREPS = [None]


def _fix_irreps_dict(d: Dict[str, Any]):
    return {k: (i if i in _SPECIAL_IRREPS else Irreps(i)) for k, i in d.items()}


def _irreps_compatible(ir1: Dict[str, Irreps], ir2: Dict[str, Irreps]):
    return all(ir1[k] == ir2[k] for k in ir1 if k in ir2)


# == general data processing ==


def to_(
    data: Type,
    device: Optional[torch.device],
) -> Type:
    """Move an AtomicDataDict to a device"""
    for k, v in data.items():
        data[k] = v.to(device=device)
    return data


def batched_from_list(data_list: List[Type]) -> Type:
    """Batch multiple AtomicDataDict.Type into one.

    Entries in the input data_list can be batched AtomicDataDict's.
    """
    # == Safety Checks ==
    num_data = len(data_list)
    if num_data == 0:
        raise RuntimeError("Cannot batch empty list of AtomicDataDict.Type")
    elif num_data == 1:
        # Short circuit
        return with_batch_(data_list[0].copy())

    # first make sure every AtomicDataDict is batched (even if trivially so)
    # with_batch_() is a no-op if data already has BATCH_KEY and NUM_NODES_KEY
    data_list = [with_batch_(data.copy()) for data in data_list]

    # now every data entry should have BATCH_KEY and NUM_NODES_KEY
    # check for inconsistent keys over the AtomicDataDicts in the list
    dict_keys = data_list[0].keys()
    assert all(
        [dict_keys == data_list[i].keys() for i in range(len(data_list))]
    ), "Found inconsistent keys across AtomicDataDict list to be batched."

    # == Batching Procedure ==
    out = {}

    # get special keys that are related to edge indices (neighborlist)
    edge_idxs = {}
    for k in dict_keys:
        if "edge_index" in k:
            edge_idxs[k] = []

    # first handle edge indices and batch properties separately
    cum_nodes: int = 0  # for edge indices
    cum_frames: int = 0  # for batch
    batches = []
    for idx in range(num_data):
        for key in edge_idxs.keys():
            edge_idxs[key].append(data_list[idx][key] + cum_nodes)
        batches.append(data_list[idx][_keys.BATCH_KEY] + cum_frames)
        cum_frames += num_frames(data_list[idx])
        cum_nodes += num_nodes(data_list[idx])

    for key in edge_idxs.keys():
        out[key] = torch.cat(edge_idxs[key], dim=1)  # (2, num_edges)

    out[_keys.BATCH_KEY] = torch.cat(batches, dim=0)

    # then handle the rest
    ignore = set(edge_idxs.keys()) | {_keys.BATCH_KEY}
    for k in dict_keys:
        # ignore these since handled previously
        if k in ignore:
            continue
        elif k in (
            _key_registry._GRAPH_FIELDS
            | _key_registry._NODE_FIELDS
            | _key_registry._EDGE_FIELDS
        ):
            out[k] = torch.cat([d[k] for d in data_list], dim=0)
        else:
            raise KeyError(f"Unregistered key {k}")

    return out


def frame_from_batched(batched_data: Type, index: int) -> Type:
    """Returns a single frame from batched data."""
    # get data with batches just in case this is called on unbatched data
    if len(batched_data.get(_keys.NUM_NODES_KEY, (None,))) == 1:
        assert index == 0
        return batched_data
    # use zero-indexing as per python norm
    N_frames = num_frames(batched_data)
    assert (
        0 <= index < N_frames
    ), f"Input data consists of {N_frames} frames so index can run from 0 to {N_frames-1} -- but given index of {index}!"
    batches = batched_data[_keys.BATCH_KEY]
    node_idx_offset = (
        0
        if index == 0
        else torch.cumsum(batched_data[_keys.NUM_NODES_KEY], 0)[index - 1]
    )
    if _keys.EDGE_INDEX_KEY in batched_data:
        edge_center_idx = batched_data[_keys.EDGE_INDEX_KEY][0]

    out = {}
    for k, v in batched_data.items():
        # short circuit tensor is empty
        if v.numel() == 0:
            continue
        if k == _keys.EDGE_INDEX_KEY:
            # special case since shape is (2, num_edges), and to remove edge index offset
            mask = torch.eq(batches[edge_center_idx], index).unsqueeze(0)
            out[k] = torch.masked_select(v, mask).view(2, -1) - node_idx_offset
        elif k in _key_registry._GRAPH_FIELDS:
            # index out relevant frame
            out[k] = v[[index]]  # to ensure batch dimension remains
        elif k in _key_registry._NODE_FIELDS:
            # mask out relevant portion
            out[k] = v[batches == index]
            if k == _keys.BATCH_KEY:
                out[k] = torch.zeros_like(out[k])
        elif k in _key_registry._EDGE_FIELDS:  # excluding edge indices
            out[k] = v[torch.eq(torch.index_select(batches, 0, edge_center_idx), index)]
        else:
            raise KeyError(f"Unregistered key {k}")

    return out


def without_nodes(data: Type, which_nodes: torch.Tensor) -> Type:
    """Returns a copy of ``data`` with ``which_nodes`` removed.

    The returned object may share references to some underlying data tensors with ``data``.

    Args:
        data (Dict[str, torch.Tensor]): ``AtomicDataDict``
        which_nodes (torch.Tensor)    : index tensor or boolean mask
    """
    device = data[_keys.POSITIONS_KEY].device
    N_nodes = num_nodes(data)

    which_nodes = torch.as_tensor(which_nodes)
    if which_nodes.dtype == torch.bool:
        node_mask = ~which_nodes
    else:
        node_mask = torch.ones(N_nodes, dtype=torch.bool, device=device)
        node_mask[which_nodes] = False
    assert node_mask.shape == (N_nodes,)

    # only keep edges where both from and to are kept
    edge_idx = data[_keys.EDGE_INDEX_KEY]
    edge_mask = node_mask[edge_idx[0]] & node_mask[edge_idx[1]]
    # create an index mapping
    new_index = torch.full((N_nodes,), -1, dtype=torch.long, device=device)
    new_index[node_mask] = torch.arange(
        node_mask.sum(), dtype=torch.long, device=device
    )

    new_dict = {}
    for k, v in data.items():
        if k == _keys.EDGE_INDEX_KEY:
            new_dict[k] = new_index[v[:, edge_mask]]
        elif k in _key_registry._GRAPH_FIELDS:
            new_dict[k] = v
        elif k in _key_registry._NODE_FIELDS:
            new_dict[k] = v[node_mask]
        elif k in _key_registry._EDGE_FIELDS:
            new_dict[k] = v[edge_mask]
        else:
            raise KeyError(f"Unregistered key {k}")

    # specially handle NUM_NODES_KEY
    # it's a graph field, so should just be copied over if present
    if _keys.NUM_NODES_KEY in new_dict:
        # if NUM_NODES_KEY is present, BATCH KEY should also be present
        assert _keys.BATCH_KEY in new_dict
        new_dict[_keys.NUM_NODES_KEY] = torch.bincount(
            new_dict[_keys.BATCH_KEY], minlength=num_frames(data)
        )

    return new_dict


# == JIT-safe "methods" for use in model code ==


def num_frames(data: Type) -> int:
    if _keys.NUM_NODES_KEY not in data:
        return 1
    else:
        return data[_keys.NUM_NODES_KEY].size(0)


def num_nodes(data: Type) -> int:
    return data[_keys.POSITIONS_KEY].size(0)


def num_edges(data: Type) -> int:
    # will not check if neighborlist is present
    return data[_keys.EDGE_INDEX_KEY].size(1)


def with_batch_(data: Type) -> Type:
    """Get batch Tensor.

    If this AtomicDataPrimitive has no ``batch``, one of all zeros will be allocated and returned.
    """
    if _keys.BATCH_KEY in data:
        assert _keys.NUM_NODES_KEY in data
        return data
    else:
        # This is a single frame, so put in info for the trivial batch
        pos = data[_keys.POSITIONS_KEY]
        # Use .expand here to avoid allocating num nodes worth of memory
        # https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
        data[_keys.BATCH_KEY] = torch.zeros(
            1, dtype=torch.long, device=pos.device
        ).expand(len(pos))
        data[_keys.NUM_NODES_KEY] = torch.full(
            (1,), len(pos), dtype=torch.long, device=pos.device
        )
        return data


# For autocomplete in IDEs, don't expose our various imports
__all__ = [
    to_,
    without_nodes,
    num_nodes,
    num_edges,
    with_batch_,
] + _keys.ALLOWED_KEYS
