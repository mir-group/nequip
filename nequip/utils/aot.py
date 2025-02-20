import torch

from nequip.nn.compile import ListInputOutputWrapper
from nequip.data import AtomicDataDict
from nequip.data._key_registry import get_dynamic_shapes
from .fx import nequip_make_fx
from .compile import prepare_model_for_compile
from .versions import check_pt2_compile_compatibility

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

    # get tolerance for sanity checks
    tol = {torch.float32: 5e-5, torch.float64: 1e-12}[model.model_dtype]

    # defensively refresh the cache
    torch._dynamo.reset()

    # === preprocess model and make_fx ===
    model_to_trace = ListInputOutputWrapper(model, input_fields, output_fields)
    model_to_trace = prepare_model_for_compile(model_to_trace, device)

    fx_model = nequip_make_fx(
        model=model_to_trace,
        data=data,
        fields=input_fields,
        seed=seed,
        check_tol=tol,
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
    aot_model = torch._inductor.aoti_load_package(out_path)
    aot_out = aot_model([data[k] for k in input_fields])
    eager_out = model(data)
    del aot_model, model
    for idx, field in enumerate(output_fields):
        assert torch.allclose(
            aot_out[idx], eager_out[field], rtol=tol, atol=tol
        ), f"AOT Inductor export eager vs export sanity check failed with MaxAbsError = {torch.max(torch.abs(aot_out[idx] - eager_out[field])).item():.6g} (tol={tol}) for field `{field}`."
    del aot_out, eager_out
    return out_path
