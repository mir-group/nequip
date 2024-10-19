import torch
import torch.nn.functional

from e3nn.o3 import Linear

from nequip.data import AtomicDataDict
from nequip.utils.scatter import scatter
from nequip.utils.misc import dtype_from_name
from nequip.utils.logger import RankedLogger
from ._graph_mixin import GraphModuleMixin

from typing import Optional, List


logger = RankedLogger(__name__, rank_zero_only=True)


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

    Note that scaling/shifting is always done (casting into) ``default_dtype``, even if ``model_dtype`` is lower precision.

    Args:
        field: the per-atom field to scale/shift.
        num_types: the number of types in the model.
        shifts: the initial shifts to use, one per atom type.
        scales: the initial scales to use, one per atom type.
        arguments_in_dataset_units: if ``True``, says that the provided shifts/scales are in dataset
            units (in which case they will be rescaled appropriately by any global rescaling later
            applied to the model); if ``False``, the provided shifts/scales will be used without modification.

            For example, if identity shifts/scales of zeros and ones are provided, this should be ``False``.
            But if scales/shifts computed from the training data are used, and are thus in dataset units,
            this should be ``True``.
        out_field: the output field; defaults to ``field``.
    """

    field: str
    out_field: str
    scales_trainble: bool
    shifts_trainable: bool
    has_scales: bool
    has_shifts: bool
    default_dtype: torch.dtype
    _use_fma: bool

    def __init__(
        self,
        field: str,
        type_names: List[str],
        shifts: Optional[List[float]],
        scales: Optional[List[float]],
        arguments_in_dataset_units: bool,
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        default_dtype: Optional[str] = None,
        irreps_in={},
    ):
        super().__init__()
        self.num_types = len(type_names)
        self.type_names = type_names
        self.field = field
        self.out_field = f"shifted_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={self.field: "0e"},  # input to shift must be a single scalar
            irreps_out={self.out_field: irreps_in[self.field]},
        )

        self.default_dtype = dtype_from_name(
            torch.get_default_dtype() if default_dtype is None else default_dtype
        )

        self.has_shifts = shifts is not None
        if shifts is not None:
            shifts = torch.as_tensor(shifts, dtype=self.default_dtype)
            if len(shifts.reshape([-1])) == 1:
                shifts = (
                    torch.ones(self.num_types, dtype=shifts.dtype, device=shifts.device)
                    * shifts
                )
            assert shifts.shape == (
                self.num_types,
            ), f"Invalid shape of shifts {shifts}"
            self.shifts_trainable = shifts_trainable
            if shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)
        else:
            self.register_buffer("shifts", torch.Tensor())

        self.has_scales = scales is not None
        if scales is not None:
            scales = torch.as_tensor(scales, dtype=self.default_dtype)
            if len(scales.reshape([-1])) == 1:
                scales = (
                    torch.ones(self.num_types, dtype=scales.dtype, device=scales.device)
                    * scales
                )
            assert scales.shape == (
                self.num_types,
            ), f"Invalid shape of scales {scales}"
            self.scales_trainable = scales_trainable
            if scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)
        else:
            self.register_buffer("scales", torch.Tensor())

        assert isinstance(arguments_in_dataset_units, bool)
        self.arguments_in_dataset_units = arguments_in_dataset_units

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if not (self.has_scales or self.has_shifts):
            return data

        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY].view(-1)
        in_field = data[self.field]
        assert len(in_field) == len(
            species_idx
        ), "in_field doesnt seem to have correct per-atom shape"

        if self.has_scales and self.has_shifts:
            # we can used an FMA for performance
            # addcmul computes
            # input + tensor1 * tensor2 elementwise
            # it will promote to widest dtype, which comes from shifts/scales
            in_field = torch.addcmul(
                torch.index_select(self.shifts, 0, species_idx).view(-1, 1),
                torch.index_select(self.scales, 0, species_idx).view(-1, 1),
                in_field,
            )
        else:
            # fallback path for mix of enabled shifts and scales
            # multiplication / addition promotes dtypes already, so no cast is needed
            # this is specifically because self.*[species_idx].view(-1, 1)
            # is never a scalar (ndim == 0), since it is always [n_atom, 1]
            if self.has_scales:
                in_field = (
                    torch.index_select(self.scales, 0, species_idx).view(-1, 1)
                    * in_field
                )
            if self.has_shifts:
                in_field = (
                    torch.index_select(self.shifts, 0, species_idx).view(-1, 1)
                    + in_field
                )
        data[self.out_field] = in_field
        return data
