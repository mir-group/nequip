import torch

from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from typing import Optional


class SaveForOutput(torch.nn.Module, GraphModuleMixin):
    """Copy a field and disconnect it from the autograd graph.

    Copy a field and disconnect it from the autograd graph, storing it under another key for inspection as part of the models output.

    Args:
        field: the field to save
        out_field: the key to put the saved copy in
    """

    field: str
    out_field: str

    def __init__(self, field: str, out_field: str, irreps_in=None):
        super().__init__()
        self._init_irreps(irreps_in=irreps_in)
        self.irreps_out[out_field] = self.irreps_in[field]
        self.field = field
        self.out_field = out_field

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.out_field] = data[self.field].detach().clone()
        return data


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
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False
