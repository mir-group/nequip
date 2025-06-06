# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
import torch.nn.functional

from e3nn.o3._linear import Linear

from nequip.data import AtomicDataDict
from nequip.data._key_registry import get_field_type
from ._graph_mixin import GraphModuleMixin
from .utils import scatter
from .model_modifier_utils import model_modifier, replace_submodules
from nequip.utils.global_dtype import _GLOBAL_DTYPE

from typing import Optional, List, Dict, Union
import warnings


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
        scales: Optional[Union[float, Dict[str, float]]] = None,
        shifts: Optional[Union[float, Dict[str, float]]] = None,
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
        # we only accept single values or dicts
        # but we previously accepted lists, so we maintain backwards compatibility for a while
        # TODO: strictly enforce only floats and dicts when the time comes
        # for now, we throw a warning to get people to migrate
        if isinstance(scales, list) or isinstance(shifts, list):
            warnings.warn(
                "\n\n!!IMPORTANT WARNING!! \nWe will stop supporting the use of lists for per-type energy scales and shifts in the next few releases. Please begin migrating to the use of dicts that map from the model's `type_names` as keys to the relevant scale or shift values. For example, the following\n\n  per_type_energy_shifts: [1, 2, 3]\n\nshould be changed to\n\n  per_type_energy_shifts:\n    C: 1\n    H: 2\n    O: 3\n\n"
            )

        # single valued case
        if isinstance(scales, float) or isinstance(scales, int):
            scales = [scales]
        if isinstance(shifts, float) or isinstance(shifts, int):
            shifts = [shifts]

        # dict case
        if isinstance(scales, dict):
            assert set(self.type_names) == set(scales.keys())
            scales = [scales[name] for name in self.type_names]
        if isinstance(shifts, dict):
            assert set(self.type_names) == set(shifts.keys())
            shifts = [shifts[name] for name in self.type_names]

        # we convert everything to lists at this point for conversion into `torch.Tensor`s
        for sc_vars in (scales, shifts):
            if sc_vars is not None:
                assert isinstance(sc_vars, list)

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
            scales = scales.reshape(-1, 1)
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
            shifts = shifts.reshape(-1, 1)
            if self.shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)
        else:
            self.register_buffer("shifts", torch.Tensor())
        self.shifts_shortcut = self.shifts.numel() == 1

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """"""
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
                scales = torch.nn.functional.embedding(types, self.scales)
        else:
            scales = self.scales  # dummy for torchscript

        if self.has_shifts:
            if self.shifts_shortcut:
                shifts = self.shifts
            else:
                shifts = torch.nn.functional.embedding(types, self.shifts)
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

    @model_modifier(persistent=True)
    @classmethod
    def modify_PerTypeScaleShift(
        cls,
        model,
        scales: Optional[Union[float, Dict[str, float]]] = None,
        shifts: Optional[Union[float, Dict[str, float]]] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
    ):
        """Modify per-type scales and shifts of a model.

        The new ``scales`` and ``shifts`` should be provided as dicts.
        The keys must correspond to the ``type_names`` registered in the model being modified, and may not include all the possible ``type_names`` of the original model.
        For example, if one uses a pretrained model with 50 atom types, and seeks to only modify 3 per-atom shifts to be consistent with a fine-tuning dataset's DFT settings, one could use

        .. code-block:: yaml

            shifts:
              C: 1.23
              H: 0.12
              O: 2.13

        In this case, the per-type atomic energy shifts of the original model will be used for every other atom type, except for atom types with the new shifts specified.

        Args:
            scales: the new per-type atomic energy scales
            shifts: the new per-type atomic energy shifts (e.g. isolated atom energies of a dataset used for fine-tuning)
            scales_trainable (bool): whether the new scales are trainable
            shifts_trainable (bool): whether the new shifts are trainable
        """

        def _helper(sc_var, vname, old):
            # get original dict values
            orig_sc_var = getattr(old, vname).detach().cpu().tolist()
            # handle special case of single-valued shortcut
            if len(orig_sc_var) != len(old.type_names):
                assert len(orig_sc_var) == 1
                orig_sc_var = orig_sc_var * len(old.type_names)
            new_sc_var = {name: val for name, val in zip(old.type_names, orig_sc_var)}
            if sc_var is not None:
                # preprocess to list if single number
                if isinstance(sc_var, float) or isinstance(sc_var, int):
                    sc_var = {name: sc_var for name in old.type_names}
                assert isinstance(sc_var, dict)
                assert all(
                    k in old.type_names for k in sc_var.keys()
                ), f"Provided `{vname}` dict keys ({sc_var.keys()}) do not match the expected type names of the model ({old.type_names})."
                # update original model's dict with new dict entries
                new_sc_var.update(sc_var)
            # if no new values provided, we default to the original model's dict entries
            return new_sc_var

        def factory(old):
            return cls(
                type_names=old.type_names,
                field=old.field,
                out_field=old.out_field,
                scales=_helper(scales, "scales", old),
                shifts=_helper(shifts, "shifts", old),
                scales_trainable=scales_trainable,
                shifts_trainable=shifts_trainable,
                irreps_in=old.irreps_in,
            )

        return replace_submodules(model, cls, factory)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} \n  scales: {_format_type_vals(self.scales.reshape(-1).tolist(), self.type_names)}\n  shifts: {_format_type_vals(self.shifts.reshape(-1).tolist(), self.type_names)}"


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
