from typing import Sequence, List, Union, Optional

import torch

from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.utils import dtype_from_name


@compile_mode("script")
class RescaleOutput(GraphModuleMixin, torch.nn.Module):
    """Wrap a model and rescale its outputs when in ``eval()`` mode.

    Note that scaling/shifting is always done (casting into) ``default_dtype``, even if ``model_dtype`` is lower precision.

    Args:
        model : GraphModuleMixin
            The model whose outputs are to be rescaled.
        scale_keys : list of keys, default []
            Which fields to rescale.
        shift_keys : list of keys, default []
            Which fields to shift after rescaling.
        scale_by : floating or Tensor, default 1.
            The scaling factor by which to multiply fields in ``scale``.
        shift_by : floating or Tensor, default 0.
            The shift to add to fields in ``shift``.
        irreps_in : dict, optional
            Extra inputs expected by this beyond those of `model`; this is only present for compatibility.
    """

    scale_keys: List[str]
    shift_keys: List[str]
    scale_trainble: bool
    rescale_trainable: bool
    _all_keys: List[str]

    has_scale: bool
    has_shift: bool

    default_dtype: torch.dtype

    def __init__(
        self,
        model: GraphModuleMixin,
        scale_keys: Union[Sequence[str], str] = [],
        shift_keys: Union[Sequence[str], str] = [],
        scale_by=None,
        shift_by=None,
        shift_trainable: bool = False,
        scale_trainable: bool = False,
        default_dtype: Optional[str] = None,
        irreps_in: dict = {},
    ):
        super().__init__()

        self.model = model
        scale_keys = [scale_keys] if isinstance(scale_keys, str) else scale_keys
        shift_keys = [shift_keys] if isinstance(shift_keys, str) else shift_keys
        all_keys = set(scale_keys).union(shift_keys)

        # Check irreps:
        for k in irreps_in:
            if k in model.irreps_in and model.irreps_in[k] != irreps_in[k]:
                raise ValueError(
                    f"For field '{k}', the provided explicit `irreps_in` ('{k}': {irreps_in[k]}) are incompataible with those of the wrapped `model` ('{k}': {model.irreps_in[k]})"
                )
        for k in all_keys:
            if k not in model.irreps_out:
                raise KeyError(
                    f"Asked to scale or shift '{k}', but '{k}' is not in the outputs of the provided `model`."
                )
        for k in shift_keys:
            if model.irreps_out[k] is not None and model.irreps_out[k].lmax > 0:
                raise ValueError(
                    f"It doesn't make sense to shift non-scalar target '{k}'."
                )

        irreps_in.update(model.irreps_in)
        self._init_irreps(irreps_in=irreps_in, irreps_out=model.irreps_out)

        self.scale_keys = list(scale_keys)
        self.shift_keys = list(shift_keys)
        self._all_keys = list(all_keys)

        self.default_dtype = dtype_from_name(
            torch.get_default_dtype() if default_dtype is None else default_dtype
        )

        self.has_scale = scale_by is not None
        self.scale_trainble = scale_trainable
        if self.has_scale:
            scale_by = torch.as_tensor(scale_by, dtype=self.default_dtype)
            if self.scale_trainble:
                self.scale_by = torch.nn.Parameter(scale_by)
            else:
                self.register_buffer("scale_by", scale_by)
        elif self.scale_trainble:
            raise ValueError(
                "Asked for a scale_trainable, but this RescaleOutput has no scaling (`scale_by = None`)"
            )
        else:
            # register dummy for TorchScript
            self.register_buffer("scale_by", torch.Tensor())

        self.has_shift = shift_by is not None
        self.rescale_trainable = shift_trainable
        if self.has_shift:
            shift_by = torch.as_tensor(shift_by, dtype=self.default_dtype)
            if self.rescale_trainable:
                self.shift_by = torch.nn.Parameter(shift_by)
            else:
                self.register_buffer("shift_by", shift_by)
        elif self.rescale_trainable:
            raise ValueError(
                "Asked for a shift_trainable, but this RescaleOutput has no shift (`shift_by = None`)"
            )
        else:
            # register dummy for TorchScript
            self.register_buffer("shift_by", torch.Tensor())

        # Finally, we tell all the modules in the model that there is rescaling
        # This allows them to update parameters, like physical constants with units,
        # that need to be scaled

        # Note that .modules() walks the full tree, including self
        for mod in self.get_inner_model().modules():
            if isinstance(mod, GraphModuleMixin):
                callback = getattr(mod, "update_for_rescale", None)
                if callable(callback):
                    # It gets the `RescaleOutput` as an argument,
                    # since that contains all relevant information
                    callback(self)

    def get_inner_model(self):
        """Get the outermost child module that is not another ``RescaleOutput``"""
        model = self.model
        while isinstance(model, RescaleOutput):
            model = model.model
        return model

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = self.model(data)
        if self.training:
            # no scaling, but still need to promote for consistent dtype behavior
            # this is hopefully a no-op in most circumstances due to a
            # preceeding PerSpecies rescale promoting to default_dtype anyway:
            for field in self._all_keys:
                data[field] = data[field].to(dtype=self.default_dtype)
        else:
            # Scale then shift
            # * and + promote dtypes by default, but not when the other
            # operand is a scalar, which `scale/shift_by` are.
            # We solve this by expanding `scale/shift_by` to tensors
            # This is free and doesn't allocate new memory on CUDA:
            # https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand
            # confirmed in PyTorch slack
            # https://pytorch.slack.com/archives/C3PDTEV8E/p1671652283801129
            if self.has_scale:
                for field in self.scale_keys:
                    v = data[field]
                    data[field] = v * self.scale_by.expand(v.shape)
            if self.has_shift:
                for field in self.shift_keys:
                    v = data[field]
                    data[field] = v + self.shift_by.expand(v.shape)
        return data

    @torch.jit.export
    def scale(
        self,
        data: AtomicDataDict.Type,
        force_process: bool = False,
    ) -> AtomicDataDict.Type:
        """Apply rescaling to ``data``, in place.

        Only processes the data if the module is in ``eval()`` mode, unless ``force_process`` is ``True``.

        Args:
            data (map-like): a dict, ``AtomicDataDict``, ``AtomicData``, ``torch_geometric.data.Batch``, or anything else dictionary-like
            force_process (bool): if ``True``, scaling will be done regardless of whether the model is in train or evaluation mode.
        Returns:
            ``data``, modified in place
        """
        data = data.copy()
        if self.training and not force_process:
            return data
        else:
            if self.has_scale:
                for field in self.scale_keys:
                    if field in data:
                        data[field] = data[field] * self.scale_by
            if self.has_shift:
                for field in self.shift_keys:
                    if field in data:
                        data[field] = data[field] + self.shift_by
            return data

    @torch.jit.export
    def unscale(
        self,
        data: AtomicDataDict.Type,
        force_process: bool = False,
    ) -> AtomicDataDict.Type:
        """Apply the inverse of the rescaling operation to ``data``, in place.

        Only processes the data if the module is in ``train()`` mode, unless ``force_process`` is ``True``.

        Args:
            data (map-like): a dict, ``AtomicDataDict``, ``AtomicData``, ``torch_geometric.data.Batch``, or anything else dictionary-like
            force_process (bool): if ``True``, unscaling will be done regardless of whether the model is in train or evaluation mode.
        Returns:
            ``data``
        """
        data = data.copy()
        if self.training or force_process:
            # To invert, -shift then divide by scale
            if self.has_shift:
                for field in self.shift_keys:
                    if field in data:
                        data[field] = data[field] - self.shift_by
            if self.has_scale:
                for field in self.scale_keys:
                    if field in data:
                        data[field] = data[field] / self.scale_by
            return data
        else:
            return data
