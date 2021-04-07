from typing import Final
import argparse
import pathlib
import logging
import yaml

import torch

from e3nn.util.jit import script

import nequip
from nequip.nn import GraphModuleMixin

CONFIG_KEY: Final[str] = "config"
NEQUIP_VERSION_KEY: Final[str] = "nequip_version"
R_MAX_KEY: Final[str] = "r_max"
N_SPECIES_KEY: Final[str] = "n_species"


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
        metadata = {CONFIG_KEY: "", NEQUIP_VERSION_KEY: ""}
        model = torch.jit.load(args.model_path, _extra_files=metadata)
        del model
        # Everything we store right now is ASCII, so decode for printing
        metadata = {k: v.decode("ascii") for k, v in metadata.items()}
        config = metadata.pop(CONFIG_KEY)
        logging.info(f"Loaded TorchScript model with metadata {metadata}")
        logging.info("Model was built with config:")
        print(config)

    elif args.command == "build":
        if not args.train_dir.is_dir():
            raise ValueError(f"{args.train_dir} is not a directory")
        # -- load model --
        model_is_jit = False
        model_path = args.train_dir / "best_model.pth"
        try:
            model = torch.jit.load(model_path)
            model_is_jit = True
            logging.info("Loaded TorchScript model")
        except RuntimeError:
            # ^ jit.load throws this when it can't find TorchScript files
            model = torch.load(model_path)
            if not isinstance(model, GraphModuleMixin):
                raise TypeError(
                    "Model contained object that wasn't a NequIP model (nequip.nn.GraphModuleMixin)"
                )
            logging.info("Loaded pickled model")

        # -- compile --
        if not model_is_jit:
            model = script(model)
            logging.info("Compiled model to TorchScript")

        # load config
        # TODO: walk module tree if config does not exist to find params?
        config_str = (args.train_dir / "config_final.yaml").read_text()
        config = yaml.load(config_str, Loader=yaml.Loader)

        # Deploy
        metadata: dict = {NEQUIP_VERSION_KEY: nequip.__version__}
        metadata[R_MAX_KEY] = str(float(config["r_max"]))
        metadata[N_SPECIES_KEY] = str(len(config["allowed_species"]))
        metadata[CONFIG_KEY] = config_str
        torch.jit.save(model, args.out_file, _extra_files=metadata)
    else:
        raise ValueError

    return


if __name__ == "__main__":
    main()
