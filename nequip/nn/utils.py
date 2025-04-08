# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from e3nn.o3._irreps import Irrep, Irreps
from nequip.data import AtomicDataDict
from typing import Optional

"""
Migrated from https://github.com/mir-group/pytorch_runstats
"""


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = Irreps(irreps_in1).simplify()
    irreps_in2 = Irreps(irreps_in2).simplify()
    ir_out = Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def with_edge_vectors_(
    data: AtomicDataDict.Type,
    with_lengths: bool = True,
    edge_index_field: str = AtomicDataDict.EDGE_INDEX_KEY,
    edge_cell_shift_field: str = AtomicDataDict.EDGE_CELL_SHIFT_KEY,
    edge_vec_field: str = AtomicDataDict.EDGE_VECTORS_KEY,
    edge_len_field: str = AtomicDataDict.EDGE_LENGTH_KEY,
) -> AtomicDataDict.Type:
    """Compute the edge displacement vectors for a graph."""
    if edge_vec_field in data:
        if with_lengths and edge_len_field not in data:
            data[edge_len_field] = torch.linalg.norm(data[edge_vec_field], dim=-1)
        return data
    else:
        # Build it dynamically
        # Note that this is backwardable, because everything (pos, cell, shifts) is Tensors.
        pos = data[AtomicDataDict.POSITIONS_KEY]
        edge_index = data[edge_index_field]
        edge_vec = torch.index_select(pos, 0, edge_index[1]) - torch.index_select(
            pos, 0, edge_index[0]
        )
        if AtomicDataDict.CELL_KEY in data:
            # ^ note that to save time we don't check that the edge_cell_shifts are trivial if no cell is provided; we just assume they are either not present or all zero.
            # NOTE: ASE cell vectors as rows convention
            cell = data[AtomicDataDict.CELL_KEY]
            edge_cell_shift = data[edge_cell_shift_field]
            if AtomicDataDict.BATCH_KEY in data:
                # treat batched cell case
                edge_indexed_batches = torch.index_select(
                    data[AtomicDataDict.BATCH_KEY], 0, edge_index[0]
                )
                # nj <- n1j <- n1j + n1i @ nij
                edge_vec = torch.baddbmm(
                    edge_vec.view(-1, 1, 3),
                    edge_cell_shift.view(-1, 1, 3),
                    torch.index_select(cell, 0, edge_indexed_batches),
                ).view(-1, 3)
                # TODO: is there a more efficient way to do the above without creating an [n_edge] and [n_edge, 3, 3] tensor?
            else:
                # if batch key absent, we assume that cell has batch dims 1,
                # so we can avoid creating the large intermediate cell tensor
                # nj <- nj + ni @ ij
                edge_vec = torch.addmm(edge_vec, edge_cell_shift, cell.view(3, 3))
        data[edge_vec_field] = edge_vec
        if with_lengths:
            data[edge_len_field] = torch.linalg.norm(edge_vec, dim=-1)
        return data
