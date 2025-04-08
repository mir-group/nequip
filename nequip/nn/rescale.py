# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin
from nequip.utils.global_dtype import _GLOBAL_DTYPE

from typing import Sequence, List, Dict, Union


@compile_mode("script")
class RescaleOutput(GraphModuleMixin, torch.nn.Module):
    """Wrap a model and rescale its outputs.

    Note that scaling is always done (casting into) ``_GLOBAL_DTYPE=torch.float64``, even if ``model_dtype`` is of lower precision.

    Args:
        model (GraphModuleMixin): model whose outputs are to be rescaled
        scale_keys (List[str])  : fields to rescale
        scale_by (float): scaling factor by which to multiply fields in ``scale_keys``
    """

    scale_keys: List[str]
    _all_keys: List[str]
    has_scale: bool

    def __init__(
        self,
        model: GraphModuleMixin,
        scale_keys: Union[Sequence[str], str],
        scale_by: float,
        irreps_in: Dict = {},
    ):
        super().__init__()

        self.model = model
        scale_keys = [scale_keys] if isinstance(scale_keys, str) else scale_keys
        all_keys = set(scale_keys)

        # Check irreps:
        for k in irreps_in:
            if k in model.irreps_in and model.irreps_in[k] != irreps_in[k]:
                raise ValueError(
                    f"For field '{k}', the provided explicit `irreps_in` ('{k}': {irreps_in[k]}) are incompataible with those of the wrapped `model` ('{k}': {model.irreps_in[k]})"
                )
        for k in all_keys:
            if k not in model.irreps_out:
                raise KeyError(
                    f"Asked to scale '{k}', but '{k}' is not in the outputs of the provided `model`."
                )

        irreps_in.update(model.irreps_in)
        self._init_irreps(irreps_in=irreps_in, irreps_out=model.irreps_out)

        self.scale_keys = list(scale_keys)
        self._all_keys = list(all_keys)

        scale_by = torch.as_tensor(scale_by, dtype=_GLOBAL_DTYPE)
        self.register_buffer("scale_by", scale_by)

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

        # Scale and + promote dtypes by default, but not when the other
        # operand is a scalar, which `scale_by` are.
        # We solve this by expanding `scale_by` to tensors
        # This is free and doesn't allocate new memory on CUDA:
        # https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand
        # confirmed in PyTorch slack
        # https://pytorch.slack.com/archives/C3PDTEV8E/p1671652283801129
        for field in self.scale_keys:
            v = data[field]
            data[field] = v * self.scale_by.expand(v.shape)
        return data
