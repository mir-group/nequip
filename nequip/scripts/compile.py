import torch

from e3nn.util.jit import script

from nequip.model import (
    override_model_compile_mode,
    ModelFromPackage,
    ModelFromCheckpoint,
)
from nequip.train.lightning import _SOLE_MODEL_KEY
from nequip.data import AtomicDataDict, compile_utils
from nequip.utils.logger import RankedLogger
from nequip.utils.compile import prepare_model_for_compile
from nequip.utils.aot import aot_export_model
from nequip.utils._global_options import _get_latest_global_options
from omegaconf import OmegaConf
import hydra

import yaml
import argparse
import pathlib
from typing import Final


# === setup logging ===
hydra.core.utils.configure_log(None)
logger = RankedLogger(__name__, rank_zero_only=True)

# hardcode a global seed for `nequip-compile`
_COMPILE_SEED: Final[int] = 1

# === Inputs and Outputs for AOT Compile ===
# standard sets of input and output fields for specific integrations

_ALLEGRO_INPUTS = [
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
]
_NEQUIP_INPUTS = _ALLEGRO_INPUTS + [
    AtomicDataDict.CELL_KEY,
    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
]

_LMP_OUTPUTS = [
    AtomicDataDict.PER_ATOM_ENERGY_KEY,
    AtomicDataDict.FORCE_KEY,
    AtomicDataDict.VIRIAL_KEY,
]

_ASE_OUTPUTS = [
    AtomicDataDict.TOTAL_ENERGY_KEY,
    AtomicDataDict.FORCE_KEY,
    # TODO: include stress?
    AtomicDataDict.STRESS_KEY,
]

_ALLEGRO_FIELDS = {"input": _ALLEGRO_INPUTS, "output": _LMP_OUTPUTS}
_NEQUIP_FIELDS = {"input": _NEQUIP_INPUTS, "output": _LMP_OUTPUTS}
_ASE_FIELDS = {"input": _NEQUIP_FIELDS, "output": _ASE_OUTPUTS}

_TARGET_DICT = {
    "pair_nequip": _NEQUIP_FIELDS,
    "pair_allegro": _ALLEGRO_FIELDS,
    "ase": _ASE_FIELDS,
}

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
        "--ckpt-path",
        help="path to a checkpoint model file",
        type=pathlib.Path,
    )

    parser.add_argument(
        "--package-path",
        help="path to a packaged model file",
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

    # args specific to export
    parser.add_argument(
        "--target",
        help="target application for AOT export (`input-fields` and `output-fields` need not be specified if `target` is specified)",
        choices=["pair_nequip", "pair_allegro", "ase"],
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

    # == device ==
    device = args.device
    device = torch.device(device)
    logger.debug(f"Using device: {device}")

    # === sanity check ===
    # == ckpt or package ==
    if (args.package_path is not None) and (args.ckpt_path is not None):
        raise RuntimeError(
            "--ckpt-path and --package-path cannot be simultaneously specified"
        )
    # convenience variable
    use_ckpt = args.ckpt_path is not None

    # == output path extension ==
    if args.mode == "torchscript":
        assert str(args.output_path).endswith(
            ".nequip.pth"
        ), "`output-path` must end with the `.nequip.pth` extension for `torchscript` compile mode"
    elif args.mode == "aotinductor":
        assert str(args.output_path).endswith(
            ".nequip.pt2"
        ), "`output-path` must end with the `.nequip.pt2` extension for `aotinductor` compile mode"

    # === set global options and load model ===
    logger.debug("Loading model ...")
    if use_ckpt:
        with override_model_compile_mode(compile_mode=None):
            model = ModelFromCheckpoint(args.ckpt_path, set_global_options=True)
    else:
        model = ModelFromPackage(args.package_path, set_global_options=True)
    model = model[args.model]
    # ^ `ModuleDict` of `GraphModel` is loaded, we then select the desired `GraphModel` (`args.model` defaults to work for single model case)

    # === combine model and global options metadata ===
    global_options_metadata = _get_latest_global_options(only_metadata_related=True)
    metadata = model.metadata.copy()
    metadata.update(global_options_metadata)
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
        return

    # === AOT Inductor ===
    if args.mode == "aotinductor":

        # === parse batch dims range ===
        batch_map = {
            "graph": _parse_bounds_to_Dim("num_frames", args.num_frames),
            "node": _parse_bounds_to_Dim("num_nodes", args.num_nodes),
            "edge": _parse_bounds_to_Dim("num_edges", args.num_edges),
        }

        # === get fields ===
        if args.target is None:
            assert (
                args.input_fields is not None and args.output_fields is not None
            ), "Either `target` or `input-fields` and `output-fields` must be provided for `aotinductor` compile mode"
            input_fields = args.input_fields
            output_fields = args.output_fields
        else:
            tdict = _TARGET_DICT[args.target]
            input_fields = tdict["input"]
            output_fields = tdict["output"]

            # make num_frames batch dims static
            # TODO: generalize this?
            if args.target in ["pair_nequip", "pair_allegro"]:
                batch_map["graph"] = torch.export.Dim.STATIC

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

        # === get data for compilation ===
        if args.data_path is not None:
            data = {}
            for k, v in torch.jit.load(args.data_path).state_dict().items():
                data[k] = v
        else:
            if use_ckpt:
                data = compile_utils.data_dict_from_checkpoint(args.ckpt_path)
            else:
                data = compile_utils.data_dict_from_package(args.package_path)
        data = AtomicDataDict.to_(data, device)

        # because of the 0/1 specialization problem, and the fact that the LAMMPS pair style requires `num_frames=1`
        # we need to augment to data to remove the `BATCH_KEY` and `NUM_NODES_KEY`
        # to take more optimized code paths
        if args.target in ["pair_nequip", "pair_allegro"]:
            data.pop(AtomicDataDict.BATCH_KEY)
            data.pop(AtomicDataDict.NUM_NODES_KEY)

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
        return


if __name__ == "__main__":
    main()
