from typing import Union, Tuple

import torch
import numpy as np
import ase
import ase.neighborlist

_USE_FREUD: bool = False
try:
    import freud

    _HAS_FREUD = True
except ImportError:
    _HAS_FREUD = False

# A type representing ASE-style periodic boundary condtions, which can be partial (the tuple case)
PBC = Union[bool, Tuple[bool, bool, bool]]


def neighbor_list_and_relative_vec(
    pos,
    r_max: float,
    self_interaction: bool = False,
    strict_self_interaction: bool = True,
    cell=None,
    pbc: PBC = False,
    _force_use_ASE: bool = False,
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
        self_interaction (bool): Whether or not to include same periodic image self-edges in the neighbor list.
        strict_self_interaction (bool): Whether to include *any* self interaction edges in the graph, even if the two
            instances of the atom are in different periodic images. Defaults to True, should be True for most applications.

    Returns:
        edge_index (torch.tensor shape [2, num_edges]): List of edges.
        edge_cell_shift (torch.tensor shape [num_edges, 3]): Relative cell shift
            vectors. Returned only if cell is not None.
        cell (torch.Tensor [3, 3]): the cell as a tensor on the correct device.
            Returned only if cell is not None.
    """
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3

    if isinstance(pos, torch.Tensor):
        temp_pos = pos.detach().numpy()
        out_dtype = pos.dtype
    else:
        temp_pos = np.asarray(pos)
        out_dtype = torch.get_default_dtype()

    # Get a cell on the CPU no matter what
    if isinstance(cell, torch.Tensor):
        temp_cell = cell.detach().cpu().numpy()
        cell_tensor = cell.to(dtype=out_dtype)
    elif cell is not None:
        temp_cell = np.asarray(cell)
        cell_tensor = torch.as_tensor(temp_cell, dtype=out_dtype)
    else:
        # ASE will "complete" this correctly.
        temp_cell = np.zeros((3, 3), dtype=temp_pos.dtype)
        cell_tensor = torch.as_tensor(temp_cell, dtype=out_dtype)

    # ASE dependent part
    temp_cell = ase.geometry.complete_cell(temp_cell)

    r_max_lt_half_box = np.all(2.0 * r_max < np.linalg.norm(temp_cell, axis=1))

    if (
        _USE_FREUD  # enabled?
        and not _force_use_ASE  # not overridden?
        and _HAS_FREUD  # available?
        and all(pbc)  # need all PBC for freud
        and strict_self_interaction  # freud can't handle without this
        and r_max_lt_half_box  # freud can't handle multiple images of same atom per center
    ):
        print("blegh")
        freud_box = freud.box.Box.from_matrix(temp_cell)
        freud_nl = (
            freud.locality.AABBQuery(freud_box, temp_pos)
            .query(
                temp_pos,
                dict(r_max=float(r_max), exclude_ii=True),
            )
            .toNeighborList()
        )
        first_idex = freud_nl.point_indices.astype(np.int64)
        second_idex = freud_nl.query_point_indices.astype(np.int64)
        shifts = freud_box.get_images(
            temp_pos[freud_nl.query_point_indices]
            + freud_box.wrap(
                temp_pos[freud_nl.point_indices]
                - temp_pos[freud_nl.query_point_indices]
            )
        )
    else:
        # Fallback to ASE
        first_idex, second_idex, shifts = ase.neighborlist.primitive_neighbor_list(
            "ijS",
            pbc,
            temp_cell,
            temp_pos,
            cutoff=float(r_max),
            self_interaction=strict_self_interaction,  # we want edges from atom to itself in different periodic images!
            use_scaled_positions=False,
        )

        # Eliminate true self-edges that don't cross periodic boundaries
        if not self_interaction:
            bad_edge = first_idex == second_idex
            bad_edge &= np.all(shifts == 0, axis=1)
            keep_edge = ~bad_edge
            if not np.any(keep_edge):
                raise ValueError(
                    "After eliminating self edges, no edges remain in this system."
                )
            first_idex = first_idex[keep_edge]
            second_idex = second_idex[keep_edge]
            shifts = shifts[keep_edge]

    # Build output into torch:
    edge_index = torch.vstack(
        (torch.LongTensor(first_idex), torch.LongTensor(second_idex))
    )

    shifts = torch.as_tensor(shifts, dtype=out_dtype)

    return edge_index, shifts, cell_tensor
