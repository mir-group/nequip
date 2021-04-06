from typing import Final
import argparse
import pathlib
import logging

import torch

from e3nn.util.jit import script

import nequip
from nequip.nn import GraphModuleMixin

R_MAX_KEY: Final[str] = "r_max"
ORIG_CONFIG_KEY: Final[str] = "orig_config"
NEQUIP_VERSION_KEY: Final[str] = "nequip_version"


def main():
    parser = argparse.ArgumentParser(
        description="Create and view information about deployed NequIP potentials. By default, deploys `model` to `outfile`."
    )
    parser.add_argument(
        "model",
        help="Saved model to process, such as `best_model.pth` from a training session.",
        type=pathlib.Path,
    )
    # we have to do something
    parser.add_argument(
        "-d",
        "--deploy-to",
        help="Output file for deployed model.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--extract-config",
        help="Extract the original configuration from model into the given file.",
        type=pathlib.Path,
    )
    args = parser.parse_args()
    if not (args.deploy_to or args.extract_config):
        parser.error(
            "At least one operation (--deploy-to, --extract-config) must be specified."
        )

    # -- load model --
    model_is_jit = False
    try:
        model = torch.jit.load(args.model)
        model_is_jit = True
        logging.info("Loaded TorchScript model")
    except RuntimeError:
        # ^ jit.load throws this when it can't find TorchScript files
        model = torch.load(args.model)
        if not isinstance(model, GraphModuleMixin):
            raise TypeError(
                "Model contained object that wasn't a NequIP model (nequip.nn.GraphModuleMixin)"
            )
        logging.info("Loaded pickled model")

    # -- compile --
    if not model_is_jit:
        model = script(model)
        logging.info("Compiled model to TorchScript")

    # -- Extract configs --
    if args.extract_config:
        if hasattr(model, ORIG_CONFIG_KEY):
            config = getattr(model, ORIG_CONFIG_KEY)
        else:
            raise KeyError("The provided model does not contain a configuration.")

        if args.extract_config.suffix == ".yaml":
            import yaml

            with open(args.extract_config, "w+") as fout:
                yaml.dump(config, fout)
        elif args.extract_config.suffix == ".json":
            import json

            with open(args.extract_config, "w+") as fout:
                json.dump(config, fout)
        else:
            raise ValueError(
                f"Don't know how to write config to file with extension `{args.extract_config.suffix}`; try .json or .yaml."
            )

    # Deploy
    if args.deploy_to:
        metadata: dict = {NEQUIP_VERSION_KEY: nequip.__version__}
        torch.jit.save(model, args.deploy_to, _extra_files=metadata)

    return


if __name__ == "__main__":
    main()
