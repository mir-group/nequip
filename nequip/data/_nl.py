# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Final, List, Optional, Union, Tuple

import numpy as np

import torch

import ase.neighborlist
from matscipy.neighbours import neighbour_list as matscipy_nl

try:
    from vesin import NeighborList as vesin_nl
except ImportError:
    pass

from . import AtomicDataDict

# use "matscipy" as default
# NOTE:
# - vesin and matscipy do not support self-interaction
# - vesin does not allow for mixed pbcs
_DEFAULT_NL_BACKEND: Final[str] = "matscipy"
assert _DEFAULT_NL_BACKEND in [
    "ase",
    "matscipy",
    "vesin",
], f"Unknown neighborlist backend = {_DEFAULT_NL_BACKEND}"


def _nl_fn(
    pos: torch.Tensor,
    r_max: float,
    cell: Optional[torch.Tensor] = None,
    pbc: Union[bool, Tuple[bool, bool, bool], torch.Tensor] = False,
    backend: str = _DEFAULT_NL_BACKEND,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Internal function to create neighbor list and neighbor vectors based on radial cutoff.

    Note: This is a private function. Users should use ``compute_neighborlist_`` instead.

    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).

    All outputs are Tensors on the same device as ``pos``; this allows future optimization of the neighbor list on the GPU.

    Args:
        pos (torch.Tensor shape [N, 3]): Positional coordinates.
        r_max (float): Radial cutoff distance for neighbor finding.
        cell (torch.Tensor shape [3, 3] or None): Cell for periodic boundary conditions. Required if any ``pbc`` is True.
        pbc (bool or 3-tuple of bool or torch.Tensor): Whether the system is periodic in each of the three cell dimensions.
        backend (str): Neighborlist backend to use ('ase', 'matscipy', or 'vesin').

    Returns:
        edge_index (torch.tensor shape [2, num_edges]): List of edges.
        edge_cell_shift (torch.tensor shape [num_edges, 3]): Relative cell shift vectors.
    """
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3
    elif isinstance(pbc, torch.Tensor):
        # convert tensor to tuple for backends (handles GPU tensors)
        pbc = tuple(pbc.detach().cpu().tolist())

    # get device and dtype from position tensor
    out_device = pos.device
    out_dtype = pos.dtype

    # NOTE: neighborlist backends (ase, matscipy, vesin) are CPU-only
    # - For training: data is typically on CPU, so no transfer occurs
    # - For inference (e.g., torchsim): GPU -> CPU transfer is expected and normal
    # TODO: potentially implement GPU-native neighborlists in the future

    # convert to numpy for neighborlist backends
    temp_pos = pos.detach().cpu().numpy()

    # get cell and complete with ASE utils
    if cell is not None:
        temp_cell = cell.detach().cpu().numpy()
    else:
        # no cell provided, check that PBC is not requested
        if pbc[0] or pbc[1] or pbc[2]:
            raise ValueError(
                "Periodic boundary conditions requested but no cell was provided."
            )
        temp_cell = np.zeros((3, 3), dtype=temp_pos.dtype)
    temp_cell = ase.geometry.complete_cell(temp_cell)

    if backend == "vesin":
        # use same mixed pbc logic as
        # https://github.com/Luthaf/vesin/blob/main/python/vesin/src/vesin/_ase.py
        if pbc[0] and pbc[1] and pbc[2]:
            periodic = True
        elif not pbc[0] and not pbc[1] and not pbc[2]:
            periodic = False
        else:
            raise ValueError(
                "different periodic boundary conditions on different axes are not supported by vesin neighborlist, use ASE or matscipy"
            )

        first_idex, second_idex, shifts = vesin_nl(
            cutoff=float(r_max), full_list=True
        ).compute(points=temp_pos, box=temp_cell, periodic=periodic, quantities="ijS")
        # vesin returns uint64
        first_idex = first_idex.astype(np.int64)
        second_idex = second_idex.astype(np.int64)

    elif backend == "matscipy":
        first_idex, second_idex, shifts = matscipy_nl(
            "ijS",
            pbc=pbc,
            cell=temp_cell,
            positions=temp_pos,
            cutoff=float(r_max),
        )
    elif backend == "ase":
        first_idex, second_idex, shifts = ase.neighborlist.primitive_neighbor_list(
            "ijS",
            pbc,
            temp_cell,
            temp_pos,
            cutoff=float(r_max),
            self_interaction=False,
            use_scaled_positions=False,
        )

    # construct output to return
    edge_index = torch.vstack(
        (torch.LongTensor(first_idex), torch.LongTensor(second_idex))
    ).to(device=out_device)
    shifts = torch.as_tensor(
        shifts,
        dtype=out_dtype,
        device=out_device,
    )
    return edge_index, shifts


def compute_neighborlist_(
    data: AtomicDataDict.Type, r_max: float, backend: str = _DEFAULT_NL_BACKEND
) -> AtomicDataDict.Type:
    """Add a neighborlist to `data` in-place.

    This can be called on already-batched data.
    """
    _data_is_batched = AtomicDataDict.BATCH_KEY in data

    to_batch: List[AtomicDataDict.Type] = []
    for idx in range(AtomicDataDict.num_frames(data)):
        # if data is unbatched, `frame_from_batched` should just be no-op
        data_per_frame = AtomicDataDict.frame_from_batched(data, idx)

        cell = data_per_frame.get(AtomicDataDict.CELL_KEY, None)
        if cell is not None:
            cell = cell.view(3, 3)  # remove batch dimension

        pbc = data_per_frame.get(AtomicDataDict.PBC_KEY, None)
        if pbc is not None:
            pbc = pbc.view(3)  # remove batch dimension

        edge_index, edge_cell_shift = _nl_fn(
            pos=data_per_frame[AtomicDataDict.POSITIONS_KEY],
            r_max=r_max,
            cell=cell,
            pbc=pbc,
            backend=backend,
        )
        # add neighborlist information
        data_per_frame[AtomicDataDict.EDGE_INDEX_KEY] = edge_index
        if (
            data.get(AtomicDataDict.CELL_KEY, None) is not None
            and edge_cell_shift is not None
        ):
            data_per_frame[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = edge_cell_shift
        to_batch.append(data_per_frame)

    # the following ensures that we preserve the batch state
    # i.e. unbatched input -> unbatched output; batched input -> batched output
    if _data_is_batched:
        # rebatch to make sure neighborlist information is in a similar batched format
        return AtomicDataDict.batched_from_list(to_batch)
    else:
        assert len(to_batch) == 1
        return to_batch[0]
