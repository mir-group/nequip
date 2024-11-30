import torch

from nequip.nn.compile import ListInputOutputWrapper
from nequip.data import AtomicDataDict, _key_registry
from .fx import nequip_make_fx
from .compile import prepare_model_for_compile

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
    test: bool = False,
    seed: int = 1,
) -> str:

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
    )

    # === perform export ===
    # == define dynamics dims ==

    # TODO (maybe): account for custom unregistered fields
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

    # === test ===
    if test:
        loaded_model = torch._inductor.aoti_load_package(out_path)
        _ = loaded_model([data[k] for k in input_fields])

    return out_path


def get_dynamic_shapes(input_fields, batch_map):
    dynamic_shapes = ()
    for field in input_fields:
        # special case edge indices (2, num_edges)
        if field == AtomicDataDict.EDGE_INDEX_KEY:
            dynamic_shapes += ({0: torch.export.Dim.STATIC, 1: batch_map["edge"]},)
        else:
            shape_dict = {
                0: batch_map[_key_registry.get_field_type(field)],
                1: torch.export.Dim.STATIC,
            }
            # NOTE that the following assumes only rank-2 cartesian tensors
            if (
                field in _key_registry._CARTESIAN_TENSOR_FIELDS
                or field == AtomicDataDict.CELL_KEY
            ):
                shape_dict.update({2: torch.export.Dim.STATIC})
            dynamic_shapes += (shape_dict,)
    return dynamic_shapes
