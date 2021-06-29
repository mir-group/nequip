import torch

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


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
