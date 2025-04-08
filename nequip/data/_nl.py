# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Final, List

import os
import warnings
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
_NEQUIP_NL: Final[str] = os.environ.get("NEQUIP_NL", "matscipy").lower()
assert _NEQUIP_NL in [
    "ase",
    "matscipy",
    "vesin",
], f"Unknown neighborlist NEQUIP_NL = {_NEQUIP_NL}"


def neighbor_list_and_relative_vec(
    pos,
    r_max,
    cell=None,
    pbc=False,
    NL=_NEQUIP_NL,
):
    """Create neighbor list and neighbor vectors based on radial cutoff.

    Create neighbor list (``edge_index``) and relative vectors
    (``edge_attr``) based on radial cutoff.

    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).

    Thus, ``edge_index`` has the same convention as the relative vectors:
    :math:`\\vec{r}_{source, target}`

    If the input positions are a tensor with ``requires_grad == True``,
    the output displacement vectors will be correctly attached to the inputs
    for autograd.

    All outputs are Tensors on the same device as ``pos``; this allows future
    optimization of the neighbor list on the GPU.

    Args:
        pos (shape [N, 3]): Positional coordinate; Tensor or numpy array. If Tensor, must be on CPU.
        r_max (float): Radial cutoff distance for neighbor finding.
        cell (numpy shape [3, 3]): Cell for periodic boundary conditions. Ignored if ``pbc == False``.
        pbc (bool or 3-tuple of bool): Whether the system is periodic in each of the three cell dimensions.

    Returns:
        edge_index (torch.tensor shape [2, num_edges]): List of edges.
        edge_cell_shift (torch.tensor shape [num_edges, 3]): Relative cell shift
            vectors. Returned only if cell is not None.
        cell (torch.Tensor [3, 3]): the cell as a tensor on the correct device.
            Returned only if cell is not None.
    """
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3

    # Either the position or the cell may be on the GPU as tensors
    if isinstance(pos, torch.Tensor):
        temp_pos = pos.detach().cpu().numpy()
        out_device = pos.device
        out_dtype = pos.dtype
    else:
        temp_pos = np.asarray(pos)
        out_device = torch.device("cpu")
        out_dtype = torch.get_default_dtype()

    # Right now, GPU tensors require a round trip
    if out_device.type != "cpu":
        warnings.warn(
            "Currently, neighborlists require a round trip to the CPU. Please pass CPU tensors if possible."
        )

    # Get a cell on the CPU no matter what
    if isinstance(cell, torch.Tensor):
        temp_cell = cell.detach().cpu().numpy()
        cell_tensor = cell.to(device=out_device, dtype=out_dtype)
    elif cell is not None:
        temp_cell = np.asarray(cell)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)
    else:
        # ASE will "complete" this correctly.
        temp_cell = np.zeros((3, 3), dtype=temp_pos.dtype)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)

    # ASE dependent part
    temp_cell = ase.geometry.complete_cell(temp_cell)

    if NL == "vesin":
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

    elif NL == "matscipy":
        first_idex, second_idex, shifts = matscipy_nl(
            "ijS",
            pbc=pbc,
            cell=temp_cell,
            positions=temp_pos,
            cutoff=float(r_max),
        )
    elif NL == "ase":
        first_idex, second_idex, shifts = ase.neighborlist.primitive_neighbor_list(
            "ijS",
            pbc,
            temp_cell,
            temp_pos,
            cutoff=float(r_max),
            self_interaction=False,
            use_scaled_positions=False,
        )

    # Build output:
    edge_index = torch.vstack(
        (torch.LongTensor(first_idex), torch.LongTensor(second_idex))
    ).to(device=out_device)

    shifts = torch.as_tensor(
        shifts,
        dtype=out_dtype,
        device=out_device,
    )
    return edge_index, shifts, cell_tensor


def compute_neighborlist_(
    data: AtomicDataDict.Type, r_max: float, **kwargs
) -> AtomicDataDict.Type:
    """Add a neighborlist to `data` in-place.

    This can be called on alredy-batched data.
    """
    to_batch: List[AtomicDataDict.Type] = []
    for idx in range(AtomicDataDict.num_frames(data)):
        data_per_frame = AtomicDataDict.frame_from_batched(data, idx)

        cell = data_per_frame.get(AtomicDataDict.CELL_KEY, None)
        if cell is not None:
            cell = cell.view(3, 3)  # remove batch dimension

        pbc = data_per_frame.get(AtomicDataDict.PBC_KEY, None)
        if pbc is not None:
            pbc = pbc.view(3)  # remove batch dimension

        edge_index, edge_cell_shift, cell = neighbor_list_and_relative_vec(
            pos=data_per_frame[AtomicDataDict.POSITIONS_KEY],
            r_max=r_max,
            cell=cell,
            pbc=pbc,
            **kwargs,
        )
        # add neighborlist information
        data_per_frame[AtomicDataDict.EDGE_INDEX_KEY] = edge_index
        if (
            data.get(AtomicDataDict.CELL_KEY, None) is not None
            and edge_cell_shift is not None
        ):
            data_per_frame[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = edge_cell_shift
        to_batch.append(data_per_frame)

    # rebatch to make sure neighborlist information is in a similar batched format
    return AtomicDataDict.batched_from_list(to_batch)
