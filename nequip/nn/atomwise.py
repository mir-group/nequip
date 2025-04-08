# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
import torch.nn.functional

from e3nn.o3._linear import Linear

from nequip.data import AtomicDataDict
from nequip.data._key_registry import get_field_type
from ._graph_mixin import GraphModuleMixin
from .utils import scatter
from nequip.utils.global_dtype import _GLOBAL_DTYPE

from typing import Optional, List, Union


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
    constant: float

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        reduce="sum",
        avg_num_atoms=None,
        irreps_in={},
    ):
        super().__init__()
        assert reduce in ("sum", "mean", "normalized_sum")
        self.constant = 1.0
        if reduce == "normalized_sum":
            assert avg_num_atoms is not None
            self.constant = float(avg_num_atoms) ** -0.5
            reduce = "sum"
        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out=(
                {self.out_field: irreps_in[self.field]}
                if self.field in irreps_in
                else {}
            ),
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        field = data[self.field]
        if AtomicDataDict.BATCH_KEY in data:
            result = scatter(
                field,
                data[AtomicDataDict.BATCH_KEY],
                dim=0,
                dim_size=AtomicDataDict.num_frames(data),
                reduce=self.reduce,
            )
        else:
            # We can significantly simplify and avoid scatters
            if self.reduce == "sum":
                result = field.sum(dim=0, keepdim=True)
            elif self.reduce == "mean":
                result = field.mean(dim=0, keepdim=True)
            else:
                assert False
        if self.constant != 1.0:
            result = result * self.constant
        data[self.out_field] = result
        return data


class PerTypeScaleShift(GraphModuleMixin, torch.nn.Module):
    """Scale and/or shift a predicted per-atom property based on (learnable) per-species/type parameters.

    Note that scaling/shifting is always done casting into the global dtype (``float64``), even if ``model_dtype`` is a lower precision.

    If a single scalar is provided for scales/shifts, a shortcut implementation is used. Otherwise, a more expensive implementation that assigns separate scales/shifts to each atom type is used.

    If scales/shifts are trainable, the more expensive implementation that assigns separate scales/shifts to each atom type is used, even if a single scalar was provided for the initialization.
    """

    field: str
    out_field: str
    has_scales: bool
    has_shifts: bool
    scales_trainble: bool
    shifts_trainable: bool

    def __init__(
        self,
        type_names: List[str],
        field: str,
        out_field: Optional[str] = None,
        scales: Optional[Union[float, List[float]]] = None,
        shifts: Optional[Union[float, List[float]]] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        irreps_in={},
    ):
        super().__init__()
        self.type_names = type_names
        self.num_types = len(type_names)

        # === fields and irreps ===
        self.field = field
        self.out_field = field if out_field is None else out_field
        assert get_field_type(self.field) == "node"
        assert get_field_type(self.out_field) == "node"

        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={self.field: "0e"},  # input to shift must be a single scalar
            irreps_out={self.out_field: irreps_in[self.field]},
        )

        # === dtype ===
        self.out_dtype = _GLOBAL_DTYPE

        # === preprocess scales and shifts ===
        if isinstance(scales, float):
            scales = [scales]
        if isinstance(shifts, float):
            shifts = [shifts]

        # === scales ===
        self.has_scales = scales is not None
        self.scales_trainable = scales_trainable
        if self.has_scales:
            scales = torch.as_tensor(scales, dtype=self.out_dtype)
            if self.scales_trainable and scales.numel() == 1:
                # effective no-op if self.num_types == 1
                scales = (
                    torch.ones(self.num_types, dtype=scales.dtype, device=scales.device)
                    * scales
                )
            assert scales.shape == (self.num_types,) or scales.numel() == 1
            if self.scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)
        else:
            self.register_buffer("scales", torch.Tensor())
        self.scales_shortcut = self.scales.numel() == 1

        # === shifts ===
        self.has_shifts = shifts is not None
        self.shifts_trainable = shifts_trainable
        if self.has_shifts:
            shifts = torch.as_tensor(shifts, dtype=self.out_dtype)
            if self.shifts_trainable and shifts.numel() == 1:
                # effective no-op if self.num_types == 1
                shifts = (
                    torch.ones(self.num_types, dtype=shifts.dtype, device=shifts.device)
                    * shifts
                )
            assert shifts.shape == (self.num_types,) or shifts.numel() == 1
            if self.shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)
        else:
            self.register_buffer("shifts", torch.Tensor())
        self.shifts_shortcut = self.shifts.numel() == 1

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # shortcut if no scales or shifts found (only dtype promotion performed)
        if not (self.has_scales or self.has_shifts):
            data[self.out_field] = data[self.field].to(self.out_dtype)
            return data

        # === set up ===
        in_field = data[self.field]
        types = data[AtomicDataDict.ATOM_TYPE_KEY].view(-1)

        if self.has_scales:
            if self.scales_shortcut:
                scales = self.scales
            else:
                scales = torch.index_select(self.scales, 0, types).view(-1, 1)
        else:
            scales = self.scales  # dummy for torchscript

        if self.has_shifts:
            if self.shifts_shortcut:
                shifts = self.shifts
            else:
                shifts = torch.index_select(self.shifts, 0, types).view(-1, 1)
        else:
            shifts = self.shifts  # dummy for torchscript

        # === explicit cast ===
        in_field = in_field.to(self.out_dtype)

        # === scale/shift ===
        if self.has_scales and self.has_shifts:
            # we can used an FMA for performance
            # addcmul computes
            # input + tensor1 * tensor2 elementwise
            # it will promote to widest dtype, which comes from shifts/scales
            in_field = torch.addcmul(shifts, scales, in_field)
        else:
            # fallback path for mix of enabled shifts and scales
            # multiplication / addition promotes dtypes already, so no cast is needed
            if self.has_scales:
                in_field = scales * in_field
            if self.has_shifts:
                in_field = shifts + in_field

        data[self.out_field] = in_field
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} \n  scales: {_format_type_vals(self.scales.tolist(), self.type_names)}\n  shifts: {_format_type_vals(self.shifts.tolist(), self.type_names)}"


def _format_type_vals(
    vals: List[float], type_names: List[str], element_formatter: str = ".6f"
) -> str:

    if vals is None or not vals:
        return f"[{', '.join(type_names)}: None]"

    if len(vals) == 1:
        return (f"[{', '.join(type_names)}: {{:{element_formatter}}}]").format(vals[0])
    elif len(vals) == len(type_names):
        return (
            "["
            + ", ".join(
                f"{{{i}[0]}}: {{{i}[1]:{element_formatter}}}" for i in range(len(vals))
            )
            + "]"
        ).format(*zip(type_names, vals))
    else:
        raise ValueError(
            f"Don't know how to format vals=`{vals}` for types {type_names} with element_formatter=`{element_formatter}`"
        )
