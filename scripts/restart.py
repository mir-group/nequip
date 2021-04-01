""" Restart previous training

Arguments: file_name config.yaml(optional)

file_name: trainer.pth from a previous training
config.yaml: any parameters that needs to be revised
"""

import logging
import torch

from sys import argv

from nequip.utils import Config, dataset_from_config, Output, load_file
from nequip.models import EnergyModel, ForceModel
from nequip.data import AtomicDataDict
from nequip.nn import RescaleOutput


def main():

    torch.set_default_dtype(torch.float32)

    # load the dictionary
    file_name = argv[1]
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
    if len(argv) > 2:
        new_config = Config.from_file(argv[2])
        config.run_name = new_config.pop("run_name", config.run_name + "_restart")
        config.update(new_config)

    # open folders
    output = Output.from_config(config)
    updated_params = output.updated_dict()
    config.update(updated_params)

    dictionary.update(dict(config))

    if config.wandb:

        import wandb
        from nequip.train.trainer_wandb import TrainerWandB

        # download from wandb set up
        wandb.init(project=config.wandb_project, config=dict(config))
        config.update(dict(wandb.config))

        wandb.run.name = config.run_name
        wandb.config.update(updated_params, allow_val_change=True)

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
