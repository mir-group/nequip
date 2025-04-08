# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from e3nn.util.jit import script

from ._workflow_utils import set_workflow_state
from ._compile_utils import COMPILE_TARGET_DICT
from nequip.model.utils import _EAGER_MODEL_KEY, _COMPILE_TIME_AOTINDUCTOR_KEY
from nequip.model.from_save import ModelFromPackage, ModelFromCheckpoint
from nequip.train.lightning import _SOLE_MODEL_KEY
from nequip.data import AtomicDataDict, compile_utils
from nequip.utils.logger import RankedLogger
from nequip.utils.compile import prepare_model_for_compile
from nequip.utils.global_state import set_global_state, get_latest_global_state
from omegaconf import OmegaConf
import hydra

import os
import yaml
import argparse
import pathlib
from typing import Final


# === setup logging ===
hydra.core.utils.configure_log(None)
logger = RankedLogger(__name__, rank_zero_only=True)

# === override model to compile ===
# we default to using the best performance compilation option, e.g. to use a custom kernel with AOT Inductor
# this env var can be toggled to always compile from eager mode
_ALWAYS_COMPILE_FROM_EAGER: Final[bool] = bool(
    int(os.getenv("NEQUIP_ALWAYS_COMPILE_FROM_EAGER", 0))
)
if _ALWAYS_COMPILE_FROM_EAGER:
    logger.info("`NEQUIP_ALWAYS_COMPILE_FROM_EAGER=1` detected")

# hardcode a global seed for `nequip-compile`
_COMPILE_SEED: Final[int] = 1

# === AOT keys ===
_AOT_METADATA_KEY = "aot_inductor.metadata"
_AOT_OUTPUT_PATH_KEY = "aot_inductor.output_path"


def _parse_bounds_to_Dim(name: str, bounds_str: str):
    if bounds_str == "static":
        return torch.export.Dim.STATIC
    else:
        min_val, max_val = bounds_str.split(",")
        return torch.export.dynamic_shapes.Dim(
            name,
            min=int(min_val),
            max=torch.inf if max_val == "inf" else int(max_val),
        )


def main(args=None):

    # === parse inputs ===
    parser = argparse.ArgumentParser(
        description="Compiles NequIP/Allegro models from checkpoint or package files."
    )

    parser.add_argument(
        "--mode",
        help="whether to use `torchscript` or `aotinductor` to compile the model",
        choices=["torchscript", "aotinductor"],
        type=str,
        required=True,
    )

    parser.add_argument(
        "--input-path",
        help="path to a checkpoint model or packaged model file",
        type=pathlib.Path,
    )

    parser.add_argument(
        "--output-path",
        help="path to write compiled model file. NOTE: a `.nequip.pth` extension is required if `--mode torchscript` is used and a `.nequip.pt2` extension is required if `--mode aotinductor` is used",
        type=pathlib.Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        help="device to run the model on",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model",
        help=f"name of model to compile -- this option is only relevant when using multiple models (default: {_SOLE_MODEL_KEY}, meant to work for the conventional single model case)",
        type=str,
        default=_SOLE_MODEL_KEY,
    )

    parser.add_argument(
        "--tf32",
        help="whether to use TF32 or not",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    # args specific to export
    parser.add_argument(
        "--target",
        help="target application for AOT export (`input-fields` and `output-fields` need not be specified if `target` is specified)",
        choices=COMPILE_TARGET_DICT.keys(),
        type=str,
        default=None,
    )

    parser.add_argument(
        "--input-fields",
        help="input fields to the model for export",
        nargs="+",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-fields",
        help="output fields of the model for export",
        nargs="+",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data-path",
        help="path to data (with realistic shapes) for compilation (if unspecified, training data will be used for compilation)",
        type=pathlib.Path,
        default=None,
    )
    # for configuring dynamic shape bounds
    parser.add_argument(
        "--num-frames",
        type=str,
        default="2,inf",
        help="bounds for num-frames in format `min,max` or `static` (default: 2,inf)",
    )
    parser.add_argument(
        "--num-edges",
        type=str,
        default="2,inf",
        help="bounds for num-edges in format `min,max` or `static` (default: 2,inf)",
    )
    parser.add_argument(
        "--num-nodes",
        type=str,
        default="2,inf",
        help="bounds for num-nodes in format `min,max` or `static` (default: 2,inf)",
    )
    parser.add_argument(
        "--inductor-configs",
        help="options for AOT Inductor (default: {})",
        nargs="+",
        type=str,
        default=[],
    )
    args = parser.parse_args(args=args)

    set_workflow_state("compile")

    # === initialize global state ===
    set_global_state(allow_tf32=args.tf32)

    # == device ==
    device = args.device
    device = torch.device(device)
    logger.info(f"Compiling for device: {device}")

    # == output path extension ==
    if args.mode == "torchscript":
        assert str(args.output_path).endswith(
            ".nequip.pth"
        ), "`output-path` must end with the `.nequip.pth` extension for `torchscript` compile mode"
    elif args.mode == "aotinductor":
        assert str(args.output_path).endswith(
            ".nequip.pt2"
        ), "`output-path` must end with the `.nequip.pt2` extension for `aotinductor` compile mode"

    # === load model ===
    # get relevant model build types (used by both checkpoint and package logic paths)
    model_compile_mode = {
        "torchscript": _EAGER_MODEL_KEY,
        "aotinductor": (
            _EAGER_MODEL_KEY
            if _ALWAYS_COMPILE_FROM_EAGER
            else _COMPILE_TIME_AOTINDUCTOR_KEY
        ),
    }[args.mode]
    logger.info(f"Loading model for compilation from {args.input_path} ...")
    # use package load path if extension matches, otherwise assume checkpoint file
    use_ckpt = not str(args.input_path).endswith(".nequip.zip")
    if use_ckpt:
        model = ModelFromCheckpoint(args.input_path, compile_mode=model_compile_mode)
    else:
        # TODO: (maybe) more robust system that goes down a priority list for packaged models to load
        # e.g. if doing `aotinductor` compile, look for `aotinductor` model first, but fallback to loading `eager` packaged model for `nequip-compile`
        # for now we just use the same `model_compile_mode`
        model = ModelFromPackage(args.input_path, compile_mode=model_compile_mode)

    model = model[args.model]
    # ^ `ModuleDict` of `GraphModel` is loaded, we then select the desired `GraphModel` (`args.model` defaults to work for single model case)

    # === combine model and global options metadata ===
    metadata = model.metadata.copy()
    metadata.update(get_latest_global_state(only_metadata_related=True))
    # ensure bool -> int for metadata
    metadata = {k: int(v) if isinstance(v, bool) else v for k, v in metadata.items()}

    logger.debug(model)

    # === TorchScript ===
    if args.mode == "torchscript":
        metadata = {k: str(v).encode("ascii") for k, v in metadata.items()}
        model = prepare_model_for_compile(model, device)
        script_model = script(model)
        torch.jit.save(script_model, args.output_path, _extra_files=metadata)
        logger.info(f"TorchScript model saved to {args.output_path}")
        set_workflow_state(None)
        return

    # === AOT Inductor ===
    if args.mode == "aotinductor":

        # === sanity check and guarded imports ===
        from nequip.utils.versions import check_pt2_compile_compatibility

        check_pt2_compile_compatibility()
        from nequip.utils.aot import aot_export_model

        # === get data for compilation ===
        if args.data_path is not None:
            # we use `torch.jit.load` to future proof for the case where C++ clients like LAMMPS would need to provide data
            data = {}
            for k, v in torch.jit.load(args.data_path).state_dict().items():
                data[k] = v
        else:
            if use_ckpt:
                data = compile_utils.data_dict_from_checkpoint(args.input_path)
            else:
                data = compile_utils.data_dict_from_package(args.input_path)
        data = AtomicDataDict.to_(data, device)

        # === parse batch dims range ===
        batch_map = {
            "graph": _parse_bounds_to_Dim("num_frames", args.num_frames),
            "node": _parse_bounds_to_Dim("num_nodes", args.num_nodes),
            "edge": _parse_bounds_to_Dim("num_edges", args.num_edges),
        }

        # === get target specific settings ===
        if args.target is None:
            assert (
                args.input_fields is not None and args.output_fields is not None
            ), "Either `target` or `input-fields` and `output-fields` must be provided for `aotinductor` compile mode"
            input_fields = args.input_fields
            output_fields = args.output_fields
        else:
            # no checks necessary here as they would have been caught by argparse earlier
            tdict = COMPILE_TARGET_DICT[args.target]
            input_fields = tdict["input"]
            output_fields = tdict["output"]
            batch_map = tdict["batch_map_settings"](batch_map)
            data = tdict["data_settings"](data)

        logger.debug(
            "Dynamic shapes:\n"
            + "\n".join(
                [
                    f"{dim.__name__:^12} range: [{dim.min}, {dim.max}]"
                    for dim in batch_map.values()
                    if dim != torch.export.Dim.STATIC
                ]
            )
        )

        # === inductor configs ===
        inductor_configs = dict(item.split("=") for item in args.inductor_configs)

        # torch will also error out later on but we can be pre-emptive
        assert _AOT_OUTPUT_PATH_KEY not in inductor_configs

        # we use the metadata key to keep our own metadata
        assert _AOT_METADATA_KEY not in inductor_configs
        metadata = {k: str(v) for k, v in metadata.items()}
        inductor_configs[_AOT_METADATA_KEY] = metadata

        logger.debug(
            "Inductor Configs:\n"
            + yaml.dump(
                OmegaConf.to_yaml(inductor_configs),
                default_flow_style=False,
                default_style="|",
            )
        )
        # === export model ===
        _ = aot_export_model(
            model=model,
            device=device,
            input_fields=input_fields,
            output_fields=output_fields,
            data=data,
            batch_map=batch_map,
            output_path=str(args.output_path),
            inductor_configs=inductor_configs,
            seed=_COMPILE_SEED,
        )
        logger.info(f"Exported model saved to {args.output_path}")
        set_workflow_state(None)
        return


if __name__ == "__main__":
    main()
