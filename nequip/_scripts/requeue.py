""" Start or automatically restart training.

Arguments: config.yaml

config.yaml: requeue=True, and workdir, root, run_name have to be unique.
"""
import logging
import argparse
from os.path import isfile

import torch

import e3nn.util.jit

from nequip.utils import Config, dataset_from_config, Output, load_file
from nequip.models import EnergyModel, ForceModel
from nequip.data import AtomicDataDict
from nequip.nn import RescaleOutput

from .train import main as train

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Start or automatically restart a NequIP training session."
    )
    parser.add_argument("config", help="configuration file")
    args = parser.parse_args(args=args)

    config = Config.from_file(
        args.config,
        defaults=dict(
            wandb=False,
            compile_model=False,
            wandb_project="NequIP",
            requeue=False,
            model_builder="nequip.models.ForceModel",
            default_dtype="float32",
        ),
    )

    assert config.requeue, "This script only works for auto requeue. Be careful!!"
    for key in ["workdir", "root", "run_name"]:
        assert isinstance(
            config[key], str
        ), f"{key} has to be defined for requeue script"

    found_restart_file = isfile(config.workdir + "/trainer.pth")
    config.restart = found_restart_file
    config.append = found_restart_file
    config.force_append = True

    # open folders
    output = Output.from_config(config)
    config.update(output.updated_dict())

    torch.set_default_dtype(
        {"float32": torch.float32, "float64": torch.float64}[config.default_dtype]
    )

    if not found_restart_file:
        config.restart = False
        config.run_time = 1
        train(config)
        return

    # load the dictionary
    # load everything from trainer.pth
    dictionary = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename=config.workdir + "/trainer.pth",
        enforced_format="torch",
    )
    for key in ["workdir", "root", "run_name"]:
        assert (
            dictionary[key] == config[key]
        ), f"{key} is not consistent with the yaml file"
    # increase max_epochs if training has hit maximum epochs
    if "progress" in dictionary:
        stop_args = dictionary["progress"].pop("stop_arg", None)
        if stop_args is not None:
            dictionary["progress"]["stop_arg"] = None
            dictionary["max_epochs"] *= 2

    dictionary["restart"] = True
    dictionary["append"] = True
    dictionary["run_time"] += 1
    config.update({k: v for k, v in dictionary.items() if k.startswith("run_")})

    if config.wandb:
        from nequip.train.trainer_wandb import TrainerWandB
        # download parameters from wandb in case of sweeping
        from nequip.utils.wandb import resume
        config = resume(config, config.restart)
        trainer = TrainerWandB.from_dict(dictionary)
    else:
        from nequip.train.trainer import Trainer
        trainer = Trainer.from_dict(dictionary)

    # Load the dataset
    dataset = dataset_from_config(config)
    logging.info(f"Successfully loaded the data set of type {dataset}...")

    # Train/test split
    trainer.set_dataset(dataset)

    # Train
    trainer.train()

    return


if __name__ == "__main__":
    main()
