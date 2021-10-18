from typing import Optional, List

import torch
import torch.nn.functional
from torch_scatter import scatter

from e3nn.o3 import Linear

from nequip.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin


class AtomwiseOperation(GraphModuleMixin, torch.nn.Module):
    def __init__(self, operation, field: str, irreps_in=None):
        super().__init__()
        self.operation = operation
        self.field = field
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={field: operation.irreps_in},
            irreps_out={field: operation.irreps_out},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.field] = self.operation(data[self.field])
        return data


class AtomwiseLinear(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: Optional[str] = None,
        irreps_in=None,
        irreps_out=None,
    ):
        super().__init__()
        self.field = field
        out_field = out_field if out_field is not None else field
        self.out_field = out_field
        if irreps_out is None:
            irreps_out = irreps_in[field]

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            irreps_out={out_field: irreps_out},
        )
        self.linear = Linear(
            irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.out_field] = self.linear(data[self.field])
        return data


class AtomwiseReduce(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self, field: str, out_field: Optional[str] = None, reduce="sum", irreps_in={}
    ):
        super().__init__()
        assert reduce in ("sum", "mean", "min", "max")
        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_in[self.field]}
            if self.field in irreps_in
            else {},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_batch(data)
        data[self.out_field] = scatter(
            data[self.field], data[AtomicDataDict.BATCH_KEY], dim=0, reduce=self.reduce
        )
        return data


class PerSpeciesScaleShift(GraphModuleMixin, torch.nn.Module):
    field: str
    out_field: str
    trainable: bool

    def __init__(
        self,
        field: str,
        num_types: int,
        out_field: Optional[str] = None,
        shifts: Optional[list] = None,
        scales: Optional[list] = None,
        trainable: bool = False,
        fixed_numerics: bool = False,
        irreps_in={},
    ):
        super().__init__()
        self.field = field
        self.out_field = f"shifted_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={self.field: "0e"},  # input to shift must be a single scalar
            irreps_out={self.out_field: irreps_in[self.field]},
        )

        shifts = (
            torch.zeros(num_types)
            if shifts is None
            else torch.as_tensor(shifts, dtype=torch.get_default_dtype())
        )
        if len(shifts.reshape([-1])) == 1:
            shifts = torch.ones(num_types) * shifts
        assert shifts.shape == (num_types,), f"Invalid shape of shifts {shifts}"
        scales = (
            torch.ones(num_types)
            if scales is None
            else torch.as_tensor(scales, dtype=torch.get_default_dtype())
        )
        if len(scales.reshape([-1])) == 1:
            scales = torch.ones(num_types) * scales
        assert scales.shape == (num_types,), f"Invalid shape of scales {scales}"

        self.trainable = trainable
        if trainable:
            self.shifts = torch.nn.Parameter(shifts)
            self.scales = torch.nn.Parameter(scales)
        else:
            self.register_buffer("shifts", shifts)
            self.register_buffer("scales", scales)
        self.fixed_numerics = fixed_numerics

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY]
        in_field = data[self.field]
        assert len(in_field) == len(
            species_idx
        ), "in_field doesnt seem to have correct per-atom shape"
        data[self.out_field] = (
            self.shifts[species_idx].view(-1, 1)
            + self.scales[species_idx].view(-1, 1) * in_field
        )
        return data

    def update_for_rescale(self, rescale_module):
        if not self.fixed_numerics and rescale_module._has_scale:
            self.scales = self.scales / rescale_module.scale_by
            self.shifts = self.shifts / rescale_module.scale_by
