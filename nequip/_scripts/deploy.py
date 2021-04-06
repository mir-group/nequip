import argparse
import pathlib

import torch

from nequip import deploy


def main():
    parser = argparse.ArgumentParser(
        description="Create and view information about deployed NequIP potentials. By default, deploys `model` to `outfile`."
    )
    parser.add_argument("model", help="Saved model to process", type=pathlib.Path)
    parser.add_argument("outfile", help="File for output", type=pathlib.Path)
    parser.add_argument(
        "--extract-config",
        help="Extract the original configuration from `model` to `outfile`.",
        action="store_true",
    )
    args = parser.parse_args()

    # attempt to load model
    model = torch.load(args.model)

    if args.extract_config:
        if hasattr(model, deploy.ORIG_CONFIG_KEY):
            config = getattr(model, deploy.ORIG_CONFIG_KEY)
        else:
            raise KeyError("The provided model does not contain a configuration.")

        if args.outfile.suffix == ".yaml":
            import yaml

            with open(args.outfile, "w+") as fout:
                yaml.dump(config, fout)
        elif args.outfile.suffix == ".json":
            import json

            with open(args.outfile, "w+") as fout:
                json.dump(config, fout)
        else:
            raise ValueError(
                f"Don't know how to write config to file with extension `{args.outfile.suffix}`; try .json or .yaml."
            )
    else:
        # Deploy
        deploy_model = deploy.make_model_deployable(model)
        deploy_model.save(args.outfile)


if __name__ == "__main__":
    main()
