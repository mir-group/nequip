# This file is a part of the `nequip` package. Please see LICENSE and README
# at the root for information on using it.
import torch
from typing import Union, Tuple, List

from nequip.nn import graph_model
from nequip.utils.versions import check_pt2_compile_compatibility
from nequip.nn.compile import DictInputOutputWrapper


def load_aotinductor_model(
    compile_path: str,
    device: Union[str, torch.device],
    input_keys: List[str],
    output_keys: List[str],
) -> Tuple[torch.nn.Module, dict]:
    """Load an AOTInductor model from a .nequip.pt2 file.

    Args:
        compile_path: path to compiled model file ending with .nequip.pt2
        device: the device to use
        input_keys: list of input field names for DictInputOutputWrapper
        output_keys: list of output field names for DictInputOutputWrapper

    Returns:
        tuple of (wrapped_model, processed_metadata)
    """
    # sanity checks
    check_pt2_compile_compatibility()

    # load compiled model
    compiled_model = torch._inductor.aoti_load_package(compile_path)
    model = DictInputOutputWrapper(compiled_model, input_keys, output_keys)

    # get and process metadata
    metadata = compiled_model.get_metadata()

    # check device compatibility
    compile_device = metadata["AOTI_DEVICE_KEY"]
    if compile_device != device:
        raise RuntimeError(
            f"`{compile_path}` was compiled for `{compile_device}` and won't work with device={device}, use device={compile_device} instead."
        )

    # process standard metadata
    metadata[graph_model.R_MAX_KEY] = float(metadata[graph_model.R_MAX_KEY])
    metadata[graph_model.TYPE_NAMES_KEY] = metadata[graph_model.TYPE_NAMES_KEY].split(
        " "
    )

    # process per-edge-type cutoffs if present
    if graph_model.PER_EDGE_TYPE_CUTOFF_KEY in metadata:
        from nequip.nn.embedding.utils import parse_per_edge_type_cutoff_metadata

        cutoff_str = metadata[graph_model.PER_EDGE_TYPE_CUTOFF_KEY]
        metadata[graph_model.PER_EDGE_TYPE_CUTOFF_KEY] = (
            parse_per_edge_type_cutoff_metadata(
                cutoff_str, metadata[graph_model.TYPE_NAMES_KEY]
            )
        )
    else:
        metadata[graph_model.PER_EDGE_TYPE_CUTOFF_KEY] = None

    return model, metadata
