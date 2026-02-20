# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from typing import Union, Tuple, List, Optional

from nequip.nn import graph_model
from nequip.utils.versions import check_pt2_compile_compatibility
from nequip.utils.aoti_metadata import (
    NEQUIP_AOTI_INPUTS_KEY,
    NEQUIP_AOTI_OUTPUTS_KEY,
    parse_aoti_keys,
)
from nequip.nn.compile import DictInputOutputWrapper


def _resolve_aot_keys(
    provided_keys: Optional[List[str]],
    metadata: dict,
    metadata_key: str,
    kind: str,
    compile_path: str,
) -> List[str]:
    """
    As of NequIP v0.17.0, we include the input and output fields in the AOTI artefact's metadata.
    Previously, we always pass it from outside, but that's brittle.
    In principle, now we can always use the metadata from the AOTI artefact to inform the input and output fields,
    but there might be failure modes where a `--target batch` model is used for the ASE intergation or a `--target ase` model is used for the torchsim integration.
    So it's safer for those integrations to also provide the input and output keys for what they expect.
    This function will check for their consistency for safety.
    """
    # for backwards compatibility since previous AOTI models don't store the key
    metadata_entry = metadata.get(metadata_key, None)

    if provided_keys is None:
        if metadata_entry is None:
            raise ValueError(
                f"{kind}_keys are required for `{compile_path}` because this AOTI artifact does not store `{kind}` keys metadata. "
                "Please pass them explicitly or recompile with a newer `nequip-compile`."
            )
        else:
            return parse_aoti_keys(metadata_entry)
    else:
        provided_keys = list(provided_keys)
        if metadata_entry is None:
            return provided_keys
        else:
            # both probided, so we check their consistency
            metadata_keys = parse_aoti_keys(metadata_entry)
            if provided_keys != metadata_keys:
                raise ValueError(
                    f"Provided {kind} keys do not match metadata for `{compile_path}`.\n"
                    f"provided={provided_keys}\n"
                    f"metadata={metadata_keys}"
                )
            return provided_keys


def load_aotinductor_model(
    compile_path: str,
    device: Union[str, torch.device],
    input_keys: Optional[List[str]] = None,
    output_keys: Optional[List[str]] = None,
) -> Tuple[torch.nn.Module, dict]:
    """Load an AOTInductor model from a .nequip.pt2 file.

    Args:
        compile_path: path to compiled model file ending with .nequip.pt2
        device: the device to use
        input_keys: optional list of expected input field names for DictInputOutputWrapper
        output_keys: optional list of expected output field names for DictInputOutputWrapper

    Returns:
        tuple of (wrapped_model, processed_metadata)
    """
    # sanity checks
    check_pt2_compile_compatibility()

    # load compiled model
    compiled_model = torch._inductor.aoti_load_package(compile_path)

    # get and process metadata
    metadata = compiled_model.get_metadata()

    input_keys = _resolve_aot_keys(
        provided_keys=input_keys,
        metadata=metadata,
        metadata_key=NEQUIP_AOTI_INPUTS_KEY,
        kind="input",
        compile_path=compile_path,
    )
    output_keys = _resolve_aot_keys(
        provided_keys=output_keys,
        metadata=metadata,
        metadata_key=NEQUIP_AOTI_OUTPUTS_KEY,
        kind="output",
        compile_path=compile_path,
    )

    model = DictInputOutputWrapper(compiled_model, input_keys, output_keys)

    # check device compatibility
    compile_device = metadata["AOTI_DEVICE_KEY"]
    if torch.device(compile_device) != torch.device(device):
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
        from nequip.nn.embedding.utils import cutoff_str_to_fulldict

        cutoff_str = metadata[graph_model.PER_EDGE_TYPE_CUTOFF_KEY]
        metadata[graph_model.PER_EDGE_TYPE_CUTOFF_KEY] = cutoff_str_to_fulldict(
            cutoff_str, metadata[graph_model.TYPE_NAMES_KEY]
        )
    else:
        metadata[graph_model.PER_EDGE_TYPE_CUTOFF_KEY] = None

    return model, metadata
