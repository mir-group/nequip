# This file is a part of the `nequip` package. Please see LICENSE and README
# at the root for information on using it.
import torch

from pathlib import Path
from typing import Union, Tuple, List, Optional

from .torchscript import load_torchscript_model
from .aotinductor import load_aotinductor_model
from nequip.utils.global_state import TF32_KEY, set_global_state


def load_compiled_model(
    compile_path: str,
    device: Union[str, torch.device],
    input_keys: Optional[List[str]] = None,
    output_keys: Optional[List[str]] = None,
) -> Tuple[torch.nn.Module, dict]:
    """Load a compiled model from either TorchScript or AOTInductor format.

    Args:
        compile_path: path to compiled model file (``.nequip.pth`` or ``.nequip.pt2``)
        device: the device to use
        input_keys: input field names for AOTInductor models (required for ``.nequip.pt2``)
        output_keys: output field names for AOTInductor models (required for ``.nequip.pt2``)

    Returns:
        tuple of (model, metadata) with model prepared for inference
    """
    compile_fname = Path(compile_path).name

    if compile_fname.endswith(".nequip.pth"):
        model, metadata = load_torchscript_model(compile_path, device)
    elif compile_fname.endswith(".nequip.pt2"):
        if input_keys is None or output_keys is None:
            raise ValueError(
                "input_keys and output_keys are required for AOTInductor models"
            )
        model, metadata = load_aotinductor_model(
            compile_path, device, input_keys, output_keys
        )
    else:
        raise ValueError(
            f"Unknown file type: {compile_fname} "
            f"(expected `*.nequip.pth` or `*.nequip.pt2`)"
        )

    # set global state from metadata
    set_global_state(
        **{
            TF32_KEY: bool(int(metadata[TF32_KEY])),
        }
    )

    # prepare model for inference
    model = model.to(device)
    model.eval()

    return model, metadata
