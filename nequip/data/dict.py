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

    # == Cartesian tensor field reshapes (ensure batch dimension present) ==

    # IMPORTANT: the following reshape logic only applies to rank-2 Cartesian tensor fields
    for k, v in data.items():
        if k in _key_registry._CARTESIAN_TENSOR_FIELDS:
            # enforce (N_frames, 3, 3) shape for graph fields, e.g. stress, virial
            # remembering to handle ASE-style 6 element Voigt order stress
            if k in _key_registry._GRAPH_FIELDS:
                err_msg = f"bad shape {v.shape} for {k} registered as a Cartesian tensor graph field---please note that only rank-2 Cartesian tensors are currently supported"
                if v.dim() == 1:  # two possibilities
                    if v.shape == (6,):
                        assert k in (
                            AtomicDataDict.STRESS_KEY,
                            AtomicDataDict.VIRIAL_KEY,
                        )
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
            elif k in _key_registry._NODE_FIELDS:
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
