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
    parser = argparse.ArgumentParser(
        description="Restart an existing NequIP training session."
    )
    parser.add_argument("session", help="trainer.pth from a previous training")
    parser.add_argument(
        "--update-config", help="File containing any config paramters to update"
    )
    args = parser.parse_args(args=args)

    torch.set_default_dtype(torch.float32)

    # load the dictionary
    file_name = args.session
    dictionary = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename=file_name,
        enforced_format="torch",
    )
    # increase max_epochs if training has hit maximum epochs
    if "progress" in dictionary:
        stop_args = dictionary["progress"].pop("stop_arg", None)
        if stop_args is not None:
            dictionary["progress"]["stop_arg"] = None
            dictionary["max_epochs"] *= 2
    config = Config(dictionary, exclude_keys=["state_dict", "progress"])
    config.run_name = config.pop("run_name", "NequIP")

    # update with new set up
    if args.update_config:
        new_config = Config.from_file(args.update_config)
        config.run_name = new_config.pop("run_name", config.run_name + "_restart")
        config.update(new_config)

    # open folders
    output = Output.from_config(config)
    config.update(output.updated_dict())

    dictionary.update(dict(config))

    if config.wandb:
        from nequip.train.trainer_wandb import TrainerWandB

        # download parameters from wandb in case of sweeping
        from nequip.utils.wandb import init_n_update

        config = init_n_update(config)

        dictionary.update(dict(config))
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
