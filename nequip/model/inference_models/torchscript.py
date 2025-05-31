# This file is a part of the `nequip` package. Please see LICENSE and README
# at the root for information on using it.
import torch

from nequip.nn import graph_model
from nequip.utils.global_state import TF32_KEY, set_global_state

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
        TF32_KEY: None,
    }
    model = torch.jit.load(compile_path, _extra_files=metadata, map_location=device)
    model = torch.jit.freeze(model)

    # process metadata
    metadata[graph_model.R_MAX_KEY] = float(metadata[graph_model.R_MAX_KEY])
    metadata[graph_model.TYPE_NAMES_KEY] = (
        metadata[graph_model.TYPE_NAMES_KEY].decode("utf-8").split(" ")
    )

    # set global state
    set_global_state(
        **{
            TF32_KEY: bool(int(metadata[TF32_KEY])),
        }
    )

    return model, metadata
