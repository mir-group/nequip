import argparse
import sys

if sys.version_info[1] >= 8:
    from typing import Final, Optional
else:
    from typing_extensions import Final, Optional
from typing import Tuple, Dict, Union

import itertools
import importlib
import pathlib
import yaml
import packaging.version
import warnings

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

import torch

from e3nn.util.jit import script

from omegaconf import OmegaConf
import hydra

from nequip.utils.misc import dtype_to_name
from nequip.utils._global_options import _set_global_options, _get_latest_global_options


CONFIG_KEY: Final[str] = "config"
NEQUIP_VERSION_KEY: Final[str] = "nequip_version"
TORCH_VERSION_KEY: Final[str] = "torch_version"
E3NN_VERSION_KEY: Final[str] = "e3nn_version"
R_MAX_KEY: Final[str] = "r_max"
PER_EDGE_TYPE_CUTOFF_KEY: Final[str] = "per_edge_type_cutoff"
N_SPECIES_KEY: Final[str] = "n_species"
TYPE_NAMES_KEY: Final[str] = "type_names"
JIT_BAILOUT_KEY: Final[str] = "_jit_bailout_depth"
JIT_FUSION_STRATEGY: Final[str] = "_jit_fusion_strategy"
TF32_KEY: Final[str] = "allow_tf32"
DEFAULT_DTYPE_KEY: Final[str] = "default_dtype"
MODEL_DTYPE_KEY: Final[str] = "model_dtype"

_ALL_METADATA_KEYS = [
    CONFIG_KEY,
    NEQUIP_VERSION_KEY,
    TORCH_VERSION_KEY,
    E3NN_VERSION_KEY,
    R_MAX_KEY,
    PER_EDGE_TYPE_CUTOFF_KEY,
    N_SPECIES_KEY,
    TYPE_NAMES_KEY,
    JIT_BAILOUT_KEY,
    JIT_FUSION_STRATEGY,
    TF32_KEY,
    DEFAULT_DTYPE_KEY,
    MODEL_DTYPE_KEY,
]


def _register_metadata_key(key: str) -> None:
    _ALL_METADATA_KEYS.append(key)


_current_metadata: Optional[dict] = None


def _set_deploy_metadata(key: str, value) -> None:
    # TODO: not thread safe but who cares?
    global _current_metadata
    if _current_metadata is None:
        pass  # not deploying right now
    elif key not in _ALL_METADATA_KEYS:
        raise KeyError(f"{key} is not a registered model deployment metadata key")
    elif key in _current_metadata:
        raise RuntimeError(f"{key} already set in the deployment metadata")
    else:
        _current_metadata[key] = value


def _compile_for_deploy(model):
    model.eval()

    if not isinstance(model, torch.jit.ScriptModule):
        model = script(model)

    return model


def load_deployed_model(
    model_path: Union[pathlib.Path, str],
    device: Union[str, torch.device] = "cpu",
    freeze: bool = True,
    set_global_options: Union[str, bool] = "warn",
) -> Tuple[torch.jit.ScriptModule, Dict[str, str]]:
    r"""Load a deployed model.

    Args:
        model_path: the path to the deployed model's ``.pth`` file.

    Returns:
        model, metadata dictionary
    """
    metadata = {k: "" for k in _ALL_METADATA_KEYS}
    try:
        model = torch.jit.load(model_path, map_location=device, _extra_files=metadata)
    except RuntimeError as e:
        raise ValueError(
            f"{model_path} does not seem to be a deployed NequIP model file. Did you forget to deploy it using `nequip-deploy`? \n\n(Underlying error: {e})"
        )
    # Confirm nequip made it
    if len(metadata[NEQUIP_VERSION_KEY]) == 0:
        if len(metadata[JIT_BAILOUT_KEY]) != 0:
            # In versions <0.6.0, there may have been a bug leading to empty "*_version"
            # metadata keys.  We can be pretty confident this is a NequIP model from
            # those versions, though, if it stored "_jit_bailout_depth"
            # https://github.com/mir-group/nequip/commit/2f43aa84542df733bbe38cb9d6cca176b0e98054
            # Likely addresses https://github.com/mir-group/nequip/issues/431
            warnings.warn(
                f"{model_path} appears to be from a older (0.5.* or earlier) version of `nequip` "
                "that pre-dates a variety of breaking changes. Please carefully check the "
                "correctness of your results for unexpected behaviour, and consider re-deploying "
                "your model using this current `nequip` installation."
            )
        else:
            raise ValueError(
                f"{model_path} does not seem to be a deployed NequIP model file"
            )
    # Confirm its TorchScript
    assert isinstance(model, torch.jit.ScriptModule)
    # Make sure we're in eval mode
    model.eval()
    # Freeze on load:
    if freeze and hasattr(model, "training"):
        # hasattr is how torch checks whether model is unfrozen
        # only freeze if already unfrozen
        model = torch.jit.freeze(model)
    # Everything we store right now is ASCII, so decode for printing
    metadata = {k: v.decode("ascii") for k, v in metadata.items()}
    # Update metadata for backward compatibility
    if metadata[DEFAULT_DTYPE_KEY] == "":
        # Default and model go together
        assert metadata[MODEL_DTYPE_KEY] == ""
        # If there isn't a dtype, it should be older than 0.6.0---but
        # this may not be reflected in the version fields (see above check)
        # So we only check if it is available:
        if len(metadata[NEQUIP_VERSION_KEY]) > 0:
            assert packaging.version.parse(
                metadata[NEQUIP_VERSION_KEY]
            ) < packaging.version.parse("0.6.0")

        # The old pre-0.6.0 defaults:
        metadata[DEFAULT_DTYPE_KEY] = "float32"
        metadata[MODEL_DTYPE_KEY] = "float32"
        warnings.warn(
            "Models deployed before v0.6.0 don't contain information about their default_dtype or model_dtype; assuming the old default of float32 for both, but this might not be right if you had explicitly set default_dtype=float64."
        )

    # Set up global settings:
    assert set_global_options in (True, False, "warn")
    if set_global_options:
        global_config_dict = {}
        global_config_dict["allow_tf32"] = bool(int(metadata[TF32_KEY]))
        # JIT strategy
        strategy = metadata.get(JIT_FUSION_STRATEGY, "")
        if strategy != "":
            strategy = [e.split(",") for e in strategy.split(";")]
            strategy = [(e[0], int(e[1])) for e in strategy]
            global_config_dict["_jit_fusion_strategy"] = strategy
        # JIT bailout
        # _set_global_options will check torch version
        jit_bailout: int = metadata.get(JIT_BAILOUT_KEY, "")
        if jit_bailout != "":
            global_config_dict["_jit_bailout_depth"] = int(jit_bailout)
        # call to actually set the global options
        _set_global_options(
            **global_config_dict,
            warn_on_override=set_global_options == "warn",
        )
    return model, metadata


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Deploy and view information about previously deployed NequIP models."
    )
    # backward compat for 3.6
    if sys.version_info[1] > 6:
        required = {"required": True}
    else:
        required = {}
    parser.add_argument("--verbose", help="log level", default="INFO", type=str)
    subparsers = parser.add_subparsers(dest="command", title="commands", **required)
    info_parser = subparsers.add_parser(
        "info", help="Get information from a deployed model file"
    )
    info_parser.add_argument(
        "model_path",
        help="Path to a deployed model file.",
        type=pathlib.Path,
    )
    info_parser.add_argument(
        "--print-config",
        help="Print the full config of the model.",
        action="store_true",
    )

    build_parser = subparsers.add_parser("build", help="Build a deployment model")
    build_parser.add_argument(
        "-ckpt_path",
        help="Path to checkpoint file",
        type=str,
    )
    build_parser.add_argument(
        "-out_file",
        help="Output file for deployed model.",
        type=pathlib.Path,
        default="deployed.pt",
    )

    # TODO: enable overriding logic when a use case arises
    build_parser.add_argument(
        "--override_model",
        help="Add or override model configuration keys from the checkpoint file. Unless you know why you need to, do not use this option.",
        type=str,
        default=None,
    )
    build_parser.add_argument(
        "--override_global_options",
        help="Add or override global_options configuration keys from the checkpoint file. Unless you know why you need to, do not use this option.",
        type=str,
        default=None,
    )

    args = parser.parse_args(args=args)

    if args.command == "build":

        # reset the global metadata dict so that model builders can fill it:
        global _current_metadata
        _current_metadata = {}

        # === load checkpoint and extract info ===
        checkpoint = torch.load(
            args.ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

        # == get global options from checkpoint and set them ==
        global_options = checkpoint["hyper_parameters"]["info_dict"]["global_options"]
        if "default_dtype" in global_options:
            global_options.pop("default_dtype")
        _set_global_options(**global_options)

        # == get metadata ==
        metadata: dict = {}

        # = versions =
        ckpt_versions = checkpoint["hyper_parameters"]["info_dict"]["versions"]

        for code, ckpt_version in ckpt_versions.items():
            version = str(importlib.import_module(code).__version__)
            # sanity check that versions for deploy matches versions from ckpt
            if ckpt_version != version:
                warnings.warn(
                    f"`{code}` versions differ between the checkpoint file ({ckpt_version}) and the current `nequip-deploy` run ({version}) -- metadata will use the version at deploy time (now), check that this is the intended behavior"
                )
            metadata[code + "_version"] = version

        # = model metadata =
        model_cfg = checkpoint["hyper_parameters"]["model_cfg"]
        metadata[MODEL_DTYPE_KEY] = dtype_to_name(model_cfg["model_dtype"])
        metadata[R_MAX_KEY] = str(model_cfg["r_max"])
        metadata[N_SPECIES_KEY] = str(len(model_cfg["type_names"]))
        metadata[TYPE_NAMES_KEY] = " ".join(model_cfg["type_names"])

        if "per_edge_type_cutoff" in model_cfg:
            from nequip.nn.embedding._edge import _process_per_edge_type_cutoff

            per_edge_type_cutoff = _process_per_edge_type_cutoff(
                model_cfg["type_names"],
                model_cfg["per_edge_type_cutoff"],
                model_cfg["r_max"],
            )
            metadata[PER_EDGE_TYPE_CUTOFF_KEY] = " ".join(
                str(e.item()) for e in per_edge_type_cutoff.view(-1)
            )

        # == global metadata ==
        global_options = _get_latest_global_options()

        metadata[JIT_BAILOUT_KEY] = str(global_options[JIT_BAILOUT_KEY])
        if (
            packaging.version.parse(torch.__version__)
            >= packaging.version.parse("1.11")
            and JIT_FUSION_STRATEGY in global_options
        ):
            metadata[JIT_FUSION_STRATEGY] = ";".join(
                "%s,%i" % e for e in global_options[JIT_FUSION_STRATEGY]
            )
        metadata[TF32_KEY] = str(int(global_options["allow_tf32"]))
        metadata[DEFAULT_DTYPE_KEY] = dtype_to_name(global_options["default_dtype"])

        # == config metadata ==
        metadata[CONFIG_KEY] = yaml.dump(
            OmegaConf.to_yaml(checkpoint["hyper_parameters"]["info_dict"]),
            default_flow_style=False,
            default_style=">",
        )

        for k, v in _current_metadata.items():
            if k in metadata:
                raise RuntimeError(f"Custom deploy key {k} was already set")
            metadata[k] = v
        _current_metadata = None
        metadata = {k: v.encode("ascii") for k, v in metadata.items()}

        # == build model from checkpoint and compile model ==
        training_module = hydra.utils.get_class(
            checkpoint["hyper_parameters"]["info_dict"]["training_module"]
        )
        lightning_module = training_module.load_from_checkpoint(args.ckpt_path)
        model = _compile_for_deploy(lightning_module.model)
        print("Compiled & optimized model.")

        torch.jit.save(model, args.out_file, _extra_files=metadata)
        return

    elif args.command == "info":
        model, metadata = load_deployed_model(
            args.model_path, set_global_options=True, freeze=False
        )
        metadata_str = "\n".join("  %s: %s" % e for e in metadata.items())
        print(f"Loaded TorchScript model with metadata:\n{metadata_str}\n")
        print(f"Model has {sum(p.numel() for p in model.parameters())} weights")
        print(
            f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable weights"
        )
        print(
            f"Model weights and buffers take {sum(p.numel() * p.element_size() for p in itertools.chain(model.parameters(), model.buffers())) / (1024 * 1024):.2f} MB"
        )
        if args.print_config:
            print(metadata.pop(CONFIG_KEY))


if __name__ == "__main__":
    main()
