from typing import List, Dict, Any, Optional

import torch

from e3nn.util._argtools import _get_device

from nequip.data import AtomicDataDict

from ._graph_mixin import GraphModuleMixin
from ._rescale import RescaleOutput


class GraphModel(GraphModuleMixin, torch.nn.Module):
    """Top-level module for any complete `nequip` model.

    Manages top-level rescaling, dtypes, and more.

    Args:

    """

    model_dtype: torch.dtype
    model_input_fields: List[str]

    _num_rescale_layers: int

    def __init__(
        self,
        model: GraphModuleMixin,
        model_dtype: Optional[torch.dtype] = None,
        model_input_fields: Dict[str, Any] = {},
    ) -> None:
        super().__init__()
        irreps_in = {
            # Things that always make sense as inputs:
            AtomicDataDict.POSITIONS_KEY: "1o",
            AtomicDataDict.EDGE_INDEX_KEY: None,
            AtomicDataDict.EDGE_CELL_SHIFT_KEY: None,
            AtomicDataDict.CELL_KEY: "1o",  # 3 of them, but still
            AtomicDataDict.BATCH_KEY: None,
            AtomicDataDict.BATCH_PTR_KEY: None,
            AtomicDataDict.ATOM_TYPE_KEY: None,
        }
        model_input_fields = AtomicDataDict._fix_irreps_dict(model_input_fields)
        assert len(set(irreps_in.keys()).intersection(model_input_fields.keys())) == 0
        irreps_in.update(model_input_fields)
        self._init_irreps(irreps_in=irreps_in, irreps_out=model.irreps_out)
        for k, irreps in model.irreps_in.items():
            if self.irreps_in.get(k, None) != irreps:
                raise RuntimeError(
                    f"Model has `{k}` in its irreps_in with irreps `{irreps}`, but `{k}` is missing from/has inconsistent irreps in model_input_fields of `{self.irreps_in.get(k, 'missing')}`"
                )
        self.model = model
        self.model_dtype = (
            model_dtype if model_dtype is not None else torch.get_default_dtype()
        )
        self.model_input_fields = list(self.irreps_in.keys())

        self._num_rescale_layers = 0
        outer_layer = self.model
        while isinstance(outer_layer, RescaleOutput):
            self._num_rescale_layers += 1
            outer_layer = outer_layer.model

    # == Rescaling ==
    @torch.jit.unused
    def all_RescaleOutputs(self) -> List[RescaleOutput]:
        """All ``RescaleOutput``s wrapping the model, in evaluation order."""
        if self._num_rescale_layers == 0:
            return []
        # we know there's at least one
        out = [self.model]
        for _ in range(self._num_rescale_layers - 1):
            out.append(out[-1].model)
        # we iterated outermost to innermost, which is opposite of evaluation order
        assert len(out) == self._num_rescale_layers
        return out[::-1]

    @torch.jit.unused
    def unscale(
        self, data: AtomicDataDict.Type, force_process: bool = False
    ) -> AtomicDataDict.Type:
        data_unscaled = data.copy()
        # we need to unscale from the outside-in:
        for layer in self.all_RescaleOutputs()[::-1]:
            data_unscaled = layer.unscale(data_unscaled, force_process=force_process)
        return data_unscaled

    @torch.jit.unused
    def scale(
        self, data: AtomicDataDict.Type, force_process: bool = False
    ) -> AtomicDataDict.Type:
        data_scaled = data.copy()
        # we need to scale from the inside out:
        for layer in self.all_RescaleOutputs():
            data_scaled = layer.scale(data_scaled, force_process=force_process)
        return data_scaled

    # == Inference ==

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # restrict the input data to allowed keys, and cast to model_dtype
        # this also prevents the model from direclty using the dict from the outside,
        # preventing weird pass-by-reference bugs
        new_data: AtomicDataDict.Type = {}
        for k, v in data.items():
            if k in self.model_input_fields:
                if v.is_floating_point():
                    v = v.to(dtype=self.model_dtype)
                new_data[k] = v
        # run the model
        data = self.model(new_data)
        return data

    # == Helpers ==

    @torch.jit.unused
    def get_device(self) -> torch.device:
        return _get_device(self)
