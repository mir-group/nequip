import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final
from typing import Tuple, Dict, Union
import argparse
import pathlib
import logging
import warnings

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

import torch

import ase.data

from e3nn.util.jit import script

from nequip.scripts.train import _set_global_options, default_config
from nequip.train import Trainer
from nequip.utils import Config
from nequip.utils.versions import check_code_version, get_config_code_versions

CONFIG_KEY: Final[str] = "config"
NEQUIP_VERSION_KEY: Final[str] = "nequip_version"
TORCH_VERSION_KEY: Final[str] = "torch_version"
E3NN_VERSION_KEY: Final[str] = "e3nn_version"
CODE_COMMITS_KEY: Final[str] = "code_commits"
R_MAX_KEY: Final[str] = "r_max"
N_SPECIES_KEY: Final[str] = "n_species"
TYPE_NAMES_KEY: Final[str] = "type_names"
JIT_BAILOUT_KEY: Final[str] = "_jit_bailout_depth"
JIT_FUSION_STRATEGY: Final[str] = "_jit_fusion_strategy"
TF32_KEY: Final[str] = "allow_tf32"

_ALL_METADATA_KEYS = [
    CONFIG_KEY,
    NEQUIP_VERSION_KEY,
    TORCH_VERSION_KEY,
    E3NN_VERSION_KEY,
    R_MAX_KEY,
    N_SPECIES_KEY,
    TYPE_NAMES_KEY,
    JIT_BAILOUT_KEY,
    JIT_FUSION_STRATEGY,
    TF32_KEY,
]


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
        # TODO: use .to()? instead of map_location
        model = torch.jit.load(model_path, map_location=device, _extra_files=metadata)
    except RuntimeError as e:
        raise ValueError(
            f"{model_path} does not seem to be a deployed NequIP model file. Did you forget to deploy it using `nequip-deploy`? \n\n(Underlying error: {e})"
        )
    # Confirm nequip made it
    if metadata[NEQUIP_VERSION_KEY] == "":
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
    # Set up global settings:
    assert set_global_options in (True, False, "warn")
    if set_global_options:
        # Set TF32 support
        # See https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if torch.cuda.is_available() and metadata[TF32_KEY] != "":
            allow_tf32 = bool(int(metadata[TF32_KEY]))
            if torch.torch.backends.cuda.matmul.allow_tf32 is not allow_tf32:
                # Update setting
                if set_global_options == "warn":
                    warnings.warn(
                        "Loaded model had a different value for allow_tf32 than was currently set; changing the GLOBAL setting!"
                    )
                torch.backends.cuda.matmul.allow_tf32 = allow_tf32
                torch.backends.cudnn.allow_tf32 = allow_tf32

        # JIT bailout
        if int(torch.__version__.split(".")[1]) >= 11:
            strategy = metadata.get(JIT_FUSION_STRATEGY, "")
            if strategy != "":
                strategy = [e.split(",") for e in strategy.split(";")]
                strategy = [(e[0], int(e[1])) for e in strategy]
            else:
                strategy = default_config["_jit_fusion_strategy"]
            # no way to get current value, so assume we are overwriting it
            if set_global_options == "warn":
                warnings.warn(
                    "Loaded model had a different value for _jit_fusion_strategy than was currently set; changing the GLOBAL setting!"
                )
            torch.jit.set_fusion_strategy(strategy)
        else:
            jit_bailout: int = metadata.get(JIT_BAILOUT_KEY, "")
            if jit_bailout == "":
                jit_bailout = default_config["_jit_bailout_depth"]
            jit_bailout = int(jit_bailout)
            # no way to get current value, so assume we are overwriting it
            if set_global_options == "warn":
                warnings.warn(
                    "Loaded model had a different value for _jit_bailout_depth than was currently set; changing the GLOBAL setting!"
                )
            torch._C._jit_set_bailout_depth(jit_bailout)
    return model, metadata


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Create and view information about deployed NequIP potentials."
    )
    # backward compat for 3.6
    if sys.version_info[1] > 6:
        required = {"required": True}
    else:
        required = {}
    subparsers = parser.add_subparsers(dest="command", title="commands", **required)
    info_parser = subparsers.add_parser(
        "info", help="Get information from a deployed model file"
    )
    info_parser.add_argument(
        "model_path",
        help="Path to a deployed model file.",
        type=pathlib.Path,
    )

    build_parser = subparsers.add_parser("build", help="Build a deployment model")
    build_parser.add_argument(
        "train_dir",
        help="Path to a working directory from a training session.",
        type=pathlib.Path,
    )
    build_parser.add_argument(
        "out_file",
        help="Output file for deployed model.",
        type=pathlib.Path,
    )

    args = parser.parse_args(args=args)

    # TODO: configurable?
    logging.basicConfig(level=logging.INFO)

    if args.command == "info":
        model, metadata = load_deployed_model(args.model_path, set_global_options=False)
        del model
        config = metadata.pop(CONFIG_KEY)
        metadata_str = "\n".join("  %s: %s" % e for e in metadata.items())
        logging.info(f"Loaded TorchScript model with metadata:\n{metadata_str}\n")
        logging.info("Model was built with config:")
        print(config)

    elif args.command == "build":
        if not args.train_dir.is_dir():
            raise ValueError(f"{args.train_dir} is not a directory")
        if args.out_file.is_dir():
            raise ValueError(
                f"{args.out_dir} is a directory, but a path to a file for the deployed model must be given"
            )

        # load config
        config = Config.from_file(str(args.train_dir / "config.yaml"))
        _set_global_options(config)

        check_code_version(config)

        # -- load model --
        model, _ = Trainer.load_model_from_training_session(
            args.train_dir, model_name="best_model.pth", device="cpu"
        )

        # -- compile --
        model = _compile_for_deploy(model)
        logging.info("Compiled & optimized model.")

        # Deploy
        metadata: dict = {}
        code_versions, code_commits = get_config_code_versions(config)
        for code, version in code_versions.items():
            metadata[code + "_version"] = version
        if len(code_commits) > 0:
            metadata[CODE_COMMITS_KEY] = ";".join(
                f"{k}={v}" for k, v in code_commits.items()
            )

        metadata[R_MAX_KEY] = str(float(config["r_max"]))
        if "allowed_species" in config:
            # This is from before the atomic number updates
            n_species = len(config["allowed_species"])
            type_names = {
                type: ase.data.chemical_symbols[atomic_num]
                for type, atomic_num in enumerate(config["allowed_species"])
            }
        else:
            # The new atomic number setup
            n_species = str(config["num_types"])
            type_names = config["type_names"]
        metadata[N_SPECIES_KEY] = str(n_species)
        metadata[TYPE_NAMES_KEY] = " ".join(type_names)

        metadata[JIT_BAILOUT_KEY] = str(config["_jit_bailout_depth"])
        metadata[TF32_KEY] = str(int(config["allow_tf32"]))
        metadata[CONFIG_KEY] = (args.train_dir / "config.yaml").read_text()

        metadata = {k: v.encode("ascii") for k, v in metadata.items()}
        torch.jit.save(model, args.out_file, _extra_files=metadata)
    else:
        raise ValueError

    return


if __name__ == "__main__":
    main()
