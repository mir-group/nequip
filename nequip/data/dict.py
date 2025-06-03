# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import numpy as np
import torch

from . import AtomicDataDict, _key_registry
from typing import Dict


def from_dict(data: Dict) -> AtomicDataDict.Type:
    """Convert a dict of data into correct dtypes/shapes according to key"""

    data = data.copy()

    # == Deal with basic variables pos, cell, pbc ==
    assert AtomicDataDict.POSITIONS_KEY in data, "At least pos must be supplied"

    cell = data.get(AtomicDataDict.CELL_KEY, None)
    pbc = data.get(AtomicDataDict.PBC_KEY, None)
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
        data[AtomicDataDict.CELL_KEY] = torch.as_tensor(
            cell, dtype=torch.get_default_dtype()
        ).reshape(-1, 3, 3)

    if pbc is not None:
        data[AtomicDataDict.PBC_KEY] = torch.as_tensor(pbc, dtype=torch.bool).reshape(
            -1, 3
        )

    # == Deal with _some_ dtype issues ==
    for k, v in data.items():
        if k in _key_registry._LONG_FIELDS:
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
    if AtomicDataDict.NUM_NODES_KEY in data:
        N_frames = AtomicDataDict.num_frames(data)
    else:
        N_frames = 1
    N_nodes = AtomicDataDict.num_nodes(data)
    if AtomicDataDict.EDGE_INDEX_KEY in data:
        N_edges = AtomicDataDict.num_edges(data)
    else:
        N_edges = None

    # == cartesian tensor shape validation ==
    # cartesian tensor fields should already be properly shaped by input parsers (e.g. ase.py)
    for k, v in data.items():
        if k in _key_registry._CARTESIAN_TENSOR_FIELDS:
            if k in _key_registry._GRAPH_FIELDS:
                # expect (N_frames, 3, 3) for graph cartesian tensors
                assert v.dim() == 3 and v.shape == (
                    N_frames,
                    3,
                    3,
                ), f"graph cartesian tensor {k} should have shape ({N_frames}, 3, 3), got {v.shape}"
            elif k in _key_registry._NODE_FIELDS:
                # expect (N_nodes, 3, 3) for node cartesian tensors
                assert v.dim() == 3 and v.shape == (
                    N_nodes,
                    3,
                    3,
                ), f"node cartesian tensor {k} should have shape ({N_nodes}, 3, 3), got {v.shape}"

    # == general shape checks ==
    for k, v in data.items():
        if len(v.shape) == 0:
            data[k] = v.unsqueeze(-1)
            v = data[k]

        if k in _key_registry._GRAPH_FIELDS:
            assert (
                v.shape[0] == N_frames
            ), f"Leading dimension of registered graph field {k} should be {N_frames}, but found shape {v.shape}."

            # NOTE: special tensors that we keep as (num_frames,)
            if v.dim() == 1 and k not in [AtomicDataDict.NUM_NODES_KEY]:
                data[k] = v.reshape((N_frames, 1))

        elif k in _key_registry._NODE_FIELDS:
            assert (
                v.shape[0] == N_nodes
            ), f"Leading dimension of registered node field {k} should be {N_nodes}, but found shape {v.shape}."

            # NOTE: special tensors that we keep as (num_nodes,)
            if v.dim() == 1 and k not in [
                AtomicDataDict.BATCH_KEY,
                AtomicDataDict.ATOMIC_NUMBERS_KEY,
                AtomicDataDict.ATOM_TYPE_KEY,
            ]:
                data[k] = v.reshape((N_nodes, 1))

        elif k in _key_registry._EDGE_FIELDS:
            if N_edges is None:
                raise ValueError(
                    f"Inconsistent data -- {k} was registered as an edge field, but no edge indices found."
                )
            else:
                assert (
                    v.shape[0] == N_edges
                ), f"Leading dimension of registered edge field {k} should be {N_edges}, but found shape {v.shape}."

    # == specific checks for basic properties (pos, cell) ==
    pos = data[AtomicDataDict.POSITIONS_KEY]
    assert pos.dim() == 2 and pos.shape[1] == 3

    if AtomicDataDict.CELL_KEY in data:
        cell = data[AtomicDataDict.CELL_KEY]
        assert cell.dim() == 3 and cell.shape == (N_frames, 3, 3)
        assert cell.dtype == pos.dtype
        pbc = data[AtomicDataDict.PBC_KEY]
        assert pbc.dim() == 2 and pbc.shape == (N_frames, 3)

    return data
