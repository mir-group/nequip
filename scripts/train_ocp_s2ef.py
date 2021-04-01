""" Train a network."""
import logging
import torch

from sys import argv

import e3nn.util.jit

from nequip.utils import Config, dataset_from_config, Output
from nequip.models import EnergyModel, ForceModel
from nequip.data import AtomicDataDict
from nequip.nn import RescaleOutput

from nequip.datasets.trajectory_lmdb import TrajectoryLmdbDataset
#from nequip.datasets.single_point_lmdb import SinglePointLmdbDataset

def main():
    config = Config.from_file(argv[1], defaults=dict(wandb=True, compile_model=False))

    torch.set_default_dtype(torch.float32)
    output = Output.from_config(config)
    config.update(output.as_dict())

    # Make the trainer
    if config.wandb:
        from nequip.train.trainer_wandb import TrainerWandB

        trainer = TrainerWandB(model=None, **dict(config))

    else:
        from nequip.train.trainer import Trainer

        trainer = Trainer(model=None, **dict(config))

    # Load the dataset
    train_dataset = TrajectoryLmdbDataset(config["dataset"][0])
    val_dataset = TrajectoryLmdbDataset(config["dataset"][1])
    logging.info(f"Successfully loaded the data set of type {train_dataset}...")

    # Train/test split
    trainer.set_dataset(train_dataset, val_dataset)

    allowed_species = list(range(1, 100))
    config.update(dict(allowed_species=allowed_species))

    # Build a model
    energy_model = EnergyModel(**dict(config))
    force_model = ForceModel(energy_model)

    logging.info("Successfully built the network...")

    # Set the trainer
    trainer.model = force_model

    # Train
    trainer.train()

    return


if __name__ == "__main__":
    main()
