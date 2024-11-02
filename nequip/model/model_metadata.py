"""
Central source of truth of model metadata fields.
"""

import torch
from omegaconf import OmegaConf
from typing import List, Final, Dict
import warnings

R_MAX_KEY: Final[str] = "r_max"
PER_EDGE_TYPE_CUTOFF_KEY: Final[str] = "per_edge_type_cutoff"
TYPE_NAMES_KEY: Final[str] = "type_names"
NUM_TYPES_KEY: Final[str] = "num_types"
MODEL_DTYPE_KEY: Final[str] = "model_dtype"

MODEL_METADATA_KEYS: List[str] = [v for k, v in globals().items() if k.endswith("_KEY")]


def model_metadata_from_config(model_config: Dict) -> Dict:
    model_metadata_dict = {}
    # manually process everything
    model_metadata_dict[MODEL_DTYPE_KEY] = model_config[MODEL_DTYPE_KEY]
    model_metadata_dict[TYPE_NAMES_KEY] = " ".join(model_config[TYPE_NAMES_KEY])
    model_metadata_dict[NUM_TYPES_KEY] = len(model_config[TYPE_NAMES_KEY])
    model_metadata_dict[R_MAX_KEY] = str(model_config[R_MAX_KEY])

    if PER_EDGE_TYPE_CUTOFF_KEY in model_config:
        from nequip.nn.embedding._edge import _process_per_edge_type_cutoff

        per_edge_type_cutoff = _process_per_edge_type_cutoff(
            model_config[TYPE_NAMES_KEY],
            model_config[PER_EDGE_TYPE_CUTOFF_KEY],
            model_config[R_MAX_KEY],
        )
        model_metadata_dict[PER_EDGE_TYPE_CUTOFF_KEY] = " ".join(
            str(e.item()) for e in per_edge_type_cutoff.view(-1)
        )
    return model_metadata_dict


def model_metadata_from_checkpoint(ckpt_path: str) -> Dict:
    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    return model_metadata_from_config(checkpoint["hyper_parameters"]["model"])


def model_metadata_from_package(package_path: str) -> Dict:
    with warnings.catch_warnings():
        # suppress torch.package TypedStorage warning
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated.*",
            category=UserWarning,
            module="torch.package.package_importer",
        )
        imp = torch.package.PackageImporter(package_path)
        metadata = imp.load_text(package="model", resource="metadata.yaml")
    metadata = OmegaConf.to_container(OmegaConf.create(metadata))
    return metadata
