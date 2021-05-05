from typing import Final, Tuple, Dict, Union
import argparse
import pathlib
import logging
import warnings
import yaml

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

import torch

from e3nn.util.jit import script

import nequip
from nequip.nn import GraphModuleMixin

CONFIG_KEY: Final[str] = "config"
NEQUIP_VERSION_KEY: Final[str] = "nequip_version"
R_MAX_KEY: Final[str] = "r_max"
N_SPECIES_KEY: Final[str] = "n_species"

_ALL_METADATA_KEYS = [CONFIG_KEY, NEQUIP_VERSION_KEY, R_MAX_KEY, N_SPECIES_KEY]


def load_deployed_model(
    model_path: Union[pathlib.Path, str], device: Union[str, torch.device] = "cpu"
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
    if metadata[NEQUIP_VERSION_KEY] == "":
        raise ValueError(
            f"{model_path} does not seem to be a deployed NequIP model file"
        )
    # Remove missing metadata
    for k in metadata:
        # TODO: some better semver based checking of versions here, or something
        if metadata[k] == "":
            warnings.warn(
                f"Metadata key `{k}` wasn't present in the saved model; this may indicate compatability issues."
            )
    # Confirm its TorchScript
    assert isinstance(model, torch.jit.ScriptModule)
    # Make sure we're in eval mode
    model.eval()
    # Everything we store right now is ASCII, so decode for printing
    metadata = {k: v.decode("ascii") for k, v in metadata.items()}
    return model, metadata


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Create and view information about deployed NequIP potentials."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, title="commands")
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
        model, metadata = load_deployed_model(args.model_path)
        del model
        config = metadata.pop(CONFIG_KEY)
        logging.info(f"Loaded TorchScript model with metadata {metadata}")
        logging.info("Model was built with config:")
        print(config)

    elif args.command == "build":
        if not args.train_dir.is_dir():
            raise ValueError(f"{args.train_dir} is not a directory")
        if args.out_file.is_dir():
            raise ValueError(
                f"{args.out_dir} is a directory, but a path to a file for the deployed model must be given"
            )
        # -- load model --
        model_is_jit = False
        model_path = args.train_dir / "best_model.pth"
        try:
            model = torch.jit.load(model_path, map_location=torch.device("cpu"))
            model_is_jit = True
            logging.info("Loaded TorchScript model")
        except RuntimeError:
            # ^ jit.load throws this when it can't find TorchScript files
            model = torch.load(model_path, map_location=torch.device("cpu"))
            if not isinstance(model, GraphModuleMixin):
                raise TypeError(
                    "Model contained object that wasn't a NequIP model (nequip.nn.GraphModuleMixin)"
                )
            logging.info("Loaded pickled model")
        model = model.to(device=torch.device("cpu"))

        # -- compile --
        if not model_is_jit:
            model = script(model)
            logging.info("Compiled model to TorchScript")

        model.eval()  # just to be sure

        model = torch.jit.freeze(model)
        logging.info("Froze TorchScript model")

        # load config
        # TODO: walk module tree if config does not exist to find params?
        config_str = (args.train_dir / "config_final.yaml").read_text()
        config = yaml.load(config_str, Loader=yaml.Loader)

        # Deploy
        metadata: dict = {NEQUIP_VERSION_KEY: nequip.__version__}
        metadata[R_MAX_KEY] = str(float(config["r_max"]))
        metadata[N_SPECIES_KEY] = str(len(config["allowed_species"]))
        metadata[CONFIG_KEY] = config_str
        metadata = {k: v.encode("ascii") for k, v in metadata.items()}
        torch.jit.save(model, args.out_file, _extra_files=metadata)
    else:
        raise ValueError

    return


if __name__ == "__main__":
    main()
