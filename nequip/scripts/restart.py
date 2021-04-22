""" Restart previous training

Arguments: file_name config.yaml(optional)

file_name: trainer.pth from a previous training
config.yaml: any parameters that needs to be revised
"""
import logging
import argparse

import torch

from nequip.utils import Config, dataset_from_config, Output, load_file


def main(args=None):
    file_name, config = parse_command_line(args)
    restart(file_name, config, mode="update")


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(
        description="Restart an existing NequIP training session."
    )
    parser.add_argument("session", help="trainer.pth from a previous training")
    parser.add_argument(
        "--update-config", help="File containing any config paramters to update"
    )
    args = parser.parse_args(args=args)

    if args.update_config:
        config = Config.from_file(args.update_config)
        config.run_name = config.pop("run_name", "NequIP") + "_restart"
    else:
        config = Config()

    return args.session, config


def restart(file_name, config, mode="update"):

    # load the dictionary
    dictionary = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename=file_name,
        enforced_format="torch",
    )

    if mode == "update":

        origin_config = Config(dictionary, exclude_keys=["state_dict", "progress"])
        origin_config.run_name = origin_config.pop("run_name", "NequIP")
        origin_config.update(config)
        del config
        config = origin_config

    elif mode == "requeue":

        for key in ["workdir", "root", "run_name"]:
            assert (
                dictionary[key] == config[key]
            ), f"{key} is not consistent with the yaml file"

        # fetch run_name, run_time and run_id
        config.update({k: v for k, v in dictionary.items() if k.startswith("run_")})

        config.run_time += 1
        dictionary["run_time"] += 1

        torch.set_default_dtype(
            {"float32": torch.float32, "float64": torch.float64}[config.default_dtype]
        )

    # open folders
    output = Output.from_config(config)
    config.update(output.updated_dict())
    dictionary.update(output.updated_dict())

    # increase max_epochs if training has hit maximum epochs
    if "progress" in dictionary:
        stop_args = dictionary["progress"].pop("stop_arg", None)
        if stop_args is not None:
            dictionary["progress"]["stop_arg"] = None
            dictionary["max_epochs"] *= 2

    if config.wandb:
        from nequip.train.trainer_wandb import TrainerWandB

        # resume wandb run
        from nequip.utils.wandb import resume

        resume(config)

        trainer = TrainerWandB.from_dict(dictionary)
    else:
        from nequip.train.trainer import Trainer

        trainer = Trainer.from_dict(dictionary)

    dataset = dataset_from_config(config)
    logging.info(f"Successfully reload the data set of type {dataset}...")

    trainer.set_dataset(dataset)
    trainer.train()

    return


if __name__ == "__main__":
    main()
