# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from nequip.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin

from typing import List, Dict, Any, Optional, Final


R_MAX_KEY: Final[str] = "r_max"
PER_EDGE_TYPE_CUTOFF_KEY: Final[str] = "per_edge_type_cutoff"
TYPE_NAMES_KEY: Final[str] = "type_names"
NUM_TYPES_KEY: Final[str] = "num_types"
MODEL_DTYPE_KEY: Final[str] = "model_dtype"


def _model_metadata_from_config(model_config: Dict[str, str]) -> Dict[str, Any]:
    model_metadata_dict = {}
    # manually process everything
    model_metadata_dict[MODEL_DTYPE_KEY] = model_config[MODEL_DTYPE_KEY]
    model_metadata_dict[TYPE_NAMES_KEY] = " ".join(model_config[TYPE_NAMES_KEY])
    model_metadata_dict[NUM_TYPES_KEY] = len(model_config[TYPE_NAMES_KEY])
    model_metadata_dict[R_MAX_KEY] = str(model_config[R_MAX_KEY])

    if PER_EDGE_TYPE_CUTOFF_KEY in model_config:
        from .embedding._edge import _process_per_edge_type_cutoff

        per_edge_type_cutoff = _process_per_edge_type_cutoff(
            model_config[TYPE_NAMES_KEY],
            model_config[PER_EDGE_TYPE_CUTOFF_KEY],
            model_config[R_MAX_KEY],
        )
        model_metadata_dict[PER_EDGE_TYPE_CUTOFF_KEY] = " ".join(
            str(e.item()) for e in per_edge_type_cutoff.view(-1)
        )
    return model_metadata_dict


class GraphModel(GraphModuleMixin, torch.nn.Module):
    """Top-level module for any complete `nequip` model.

    Manages top-level rescaling, dtypes, and more.

    Args:
        model (GraphModuleMixin): model to wrap
        model_input_fields (Dict[str, Any]): input fields and their irreps
    """

    model_input_fields: List[str]
    is_graph_model: Final[bool] = True
    # ^ to identify `GraphModel` types from `nequip-package`d models (see https://pytorch.org/docs/stable/package.html#torch-package-sharp-edges)

    def __init__(
        self,
        model: GraphModuleMixin,
        model_config: Optional[Dict[str, str]] = None,
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
            AtomicDataDict.NUM_NODES_KEY: None,
            AtomicDataDict.ATOM_TYPE_KEY: None,
        }
        model_input_fields = AtomicDataDict._fix_irreps_dict(model_input_fields)
        irreps_in.update(model_input_fields)
        self._init_irreps(irreps_in=irreps_in, irreps_out=model.irreps_out)
        for k, irreps in model.irreps_in.items():
            if self.irreps_in.get(k, None) != irreps:
                raise RuntimeError(
                    f"Model has `{k}` in its irreps_in with irreps `{irreps}`, but `{k}` is missing from/has inconsistent irreps in model_input_fields of `{self.irreps_in.get(k, 'missing')}`"
                )
        self.model = model
        self.model_input_fields = list(self.irreps_in.keys())

        # the following logic is for backward compatibility and to simplify unittests
        self.model_dtype = torch.get_default_dtype()
        self.metadata = {}
        self.type_names = []
        if model_config is not None:
            self.metadata = _model_metadata_from_config(model_config)
            self.type_names = self.metadata[TYPE_NAMES_KEY].split(" ")
            model_dtype = {"float32": torch.float32, "float64": torch.float64}[
                self.metadata[MODEL_DTYPE_KEY]
            ]
            assert self.model_dtype == model_dtype

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # restrict the input data to allowed keys to prevent the model from directly using the dict from the outside,
        # preventing weird pass-by-reference bugs
        new_data: AtomicDataDict.Type = {}
        for k in self.model_input_fields:
            if k in data:
                new_data[k] = data[k]
        return self.model(new_data)
