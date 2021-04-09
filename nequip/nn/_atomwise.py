from typing import Optional, List

import torch
import torch.nn.functional
from torch_scatter import scatter

from e3nn.o3 import Linear

from nequip.data import AtomicDataDict
from nequip.utils.batch_ops import bincount
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


class PerSpeciesShift(GraphModuleMixin, torch.nn.Module):
    enabled: bool
    field: str
    out_field: str
    trainable: bool

    def __init__(
        self,
        field: str,
        allowed_species: List[int],
        out_field: Optional[str] = None,
        shifts: Optional[list] = None,
        scales: Optional[list] = None,
        trainable: bool = False,
        enabled: bool = False,
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

        self.enabled = enabled
        self.trainable = trainable

        shifts = (
            torch.zeros(len(allowed_species))
            if shifts is None
            else torch.as_tensor(shifts, dtype=torch.get_default_dtype())
        )
        assert shifts.shape == (
            len(allowed_species),
        ), f"Invalid shape of shifts {shifts}"
        scales = (
            torch.ones(len(allowed_species))
            if scales is None
            else torch.as_tensor(scales, dtype=torch.get_default_dtype())
        )
        assert scales.shape == (
            len(allowed_species),
        ), f"Invalid shape of scales {scales}"

        if trainable:
            self.shifts = torch.nn.Parameter(shifts)
            self.scales = torch.nn.Parameter(scales)
        else:
            self.register_buffer("shifts", shifts)
            self.register_buffer("scales", scales)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.enabled:
            species_idx = data[AtomicDataDict.SPECIES_INDEX_KEY]
            data[self.out_field] = (
                self.shifts[species_idx] + self.scales[species_idx] * data[self.field]
            )
        else:
            data[self.out_field] = data[self.field]
        return data
