# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from nequip.nn.compile import ListInputOutputWrapper, DictInputOutputWrapper
from nequip.data import AtomicDataDict
from nequip.data._key_registry import get_dynamic_shapes
from .fx import nequip_make_fx
from .compile import prepare_model_for_compile
from .versions import check_pt2_compile_compatibility
from .dtype import test_model_output_similarity_by_dtype, _pt2_compile_error_message

from typing import List, Dict, Union, Any


def aot_export_model(
    model: torch.nn.Module,
    device: Union[str, torch.device],
    input_fields: List[str],
    output_fields: List[str],
    data: AtomicDataDict.Type,
    batch_map: Dict[str, torch.export.dynamic_shapes.Dim],
    output_path: str,
    inductor_configs: Dict[str, Any] = {},
    seed: int = 1,
) -> str:
    # === torch version check ===
    check_pt2_compile_compatibility()

    # defensively refresh the cache
    torch._dynamo.reset()

    # === preprocess model and make_fx ===
    model_to_trace = ListInputOutputWrapper(model, input_fields, output_fields)
    model_to_trace = prepare_model_for_compile(model_to_trace, device)

    fx_model = nequip_make_fx(
        model=model_to_trace,
        data={k: data[k] for k in input_fields},
        fields=input_fields,
        seed=seed,
    )

    # === perform export ===
    # == define dynamics dims ==
    dynamic_shapes = get_dynamic_shapes(input_fields, batch_map)

    # == export ==
    exported = torch.export.export(
        fx_model,
        (*[data[k] for k in input_fields],),
        dynamic_shapes=dynamic_shapes,
    )
    # NOTE: the following requires PyTorch 2.6
    out_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=output_path,
        inductor_configs=inductor_configs,
    )
    assert out_path == output_path

    # === sanity check ===
    aot_model = DictInputOutputWrapper(
        torch._inductor.aoti_load_package(out_path),
        input_fields,
        output_fields,
    )
    test_model_output_similarity_by_dtype(
        aot_model,
        model,
        {k: data[k] for k in input_fields},
        model.model_dtype,
        fields=output_fields,
        error_message=_pt2_compile_error_message,
    )
    del aot_model

    return out_path
