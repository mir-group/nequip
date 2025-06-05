# This file is a part of the `nequip` package. Please see LICENSE and README
# at the root for information on using it.
import torch
from e3nn.util.jit import script

from nequip.nn import graph_model
from nequip.utils.compile import prepare_model_for_compile
from nequip.utils.global_state import TF32_KEY

from typing import Union, Tuple


def load_torchscript_model(
    compile_path: str,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.nn.Module, dict]:
    """Load a torchscript model from a .nequip.pth file.

    Args:
        compile_path (str): path to compiled model file ending with .nequip.pth
        device (Union[str, torch.device]): the device to use
    """
    # load model with metadata
    metadata = {
        graph_model.R_MAX_KEY: None,
        graph_model.TYPE_NAMES_KEY: None,
        graph_model.PER_EDGE_TYPE_CUTOFF_KEY: None,
        TF32_KEY: None,
    }
    model = torch.jit.load(compile_path, _extra_files=metadata, map_location=device)
    model = torch.jit.freeze(model)

    # process metadata
    metadata[graph_model.R_MAX_KEY] = float(metadata[graph_model.R_MAX_KEY])
    metadata[graph_model.TYPE_NAMES_KEY] = (
        metadata[graph_model.TYPE_NAMES_KEY].decode("utf-8").split(" ")
    )

    # process per-edge-type cutoffs if present
    if metadata[graph_model.PER_EDGE_TYPE_CUTOFF_KEY] is not None:
        from nequip.nn.embedding.utils import parse_per_edge_type_cutoff_metadata

        cutoff_str = metadata[graph_model.PER_EDGE_TYPE_CUTOFF_KEY].decode("utf-8")
        metadata[graph_model.PER_EDGE_TYPE_CUTOFF_KEY] = (
            parse_per_edge_type_cutoff_metadata(
                cutoff_str, metadata[graph_model.TYPE_NAMES_KEY]
            )
        )

    return model, metadata


def save_torchscript_model(
    model: torch.nn.Module,
    metadata: dict,
    output_path: str,
    device: Union[str, torch.device],
) -> None:
    """Save a model as a torchscript .nequip.pth file.

    Args:
        model: model to save
        metadata: metadata dictionary to save with the model
        output_path: path to save the compiled model
        device: device to prepare model on
    """
    # encode metadata for torchscript
    encoded_metadata = {k: str(v).encode("ascii") for k, v in metadata.items()}

    # prepare and script model
    model = prepare_model_for_compile(model, device)
    script_model = script(model)

    # save with metadata
    torch.jit.save(script_model, output_path, _extra_files=encoded_metadata)
