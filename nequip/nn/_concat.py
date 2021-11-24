from typing import List

import torch

from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


class Concat(GraphModuleMixin, torch.nn.Module):
    """Concatenate multiple fields into one."""

    def __init__(self, in_fields: List[str], out_field: str, irreps_in={}):
        super().__init__()
        self.in_fields = list(in_fields)
        self.out_field = out_field
        self._init_irreps(irreps_in=irreps_in, required_irreps_in=self.in_fields)
        self.irreps_out[self.out_field] = sum(
            (self.irreps_in[k] for k in self.in_fields), o3.Irreps()
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.out_field] = torch.cat([data[k] for k in self.in_fields], dim=-1)
        return data
