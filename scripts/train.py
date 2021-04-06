""" Train a network."""
import logging
import torch

from sys import argv

import e3nn.util.jit

from nequip.utils import Config, dataset_from_config, Output
from nequip.models import EnergyModel, ForceModel
from nequip.data import AtomicDataDict
from nequip.nn import RescaleOutput
from nequip.deploy import ORIG_CONFIG_KEY


def main():

    config = Config.from_file(
        argv[1], defaults=dict(wandb=False, compile_model=False, wandb_project="NequIP")
    )

    torch.set_default_dtype(torch.float32)
    output = Output.from_config(config)
    config.update(output.updated_dict())

    # Make the trainer
    if config.wandb:
        import wandb
        from nequip.train.trainer_wandb import TrainerWandB

        # download parameters from wandb in case of sweeping
        from nequip.utils.wandb import init_n_update

        config = init_n_update(config)

        trainer = TrainerWandB(model=None, **dict(config))
    else:
        from nequip.train.trainer import Trainer

        trainer = Trainer(model=None, **dict(config))

    # Load the dataset
    dataset = dataset_from_config(config)
    logging.info(f"Successfully loaded the data set of type {dataset}...")

    # Train/test split
    trainer.set_dataset(dataset)

    # Get statistics of training dataset
    (
        (forces_std,),
        (energies_mean, energies_std),
        (allowed_species, Z_count),
    ) = trainer.dataset_train.statistics(
        fields=[
            AtomicDataDict.FORCE_KEY,
            AtomicDataDict.TOTAL_ENERGY_KEY,
            AtomicDataDict.ATOMIC_NUMBERS_KEY,
        ],
        modes=["rms", "mean_std", "count"],
    )

    config.update(dict(allowed_species=allowed_species))

    # Build a model
    energy_model = EnergyModel(**dict(config))
    force_model = ForceModel(energy_model)

    logging.info("Successfully built the network...")

    core_model = RescaleOutput(
        model=force_model,
        scale_keys=[
            AtomicDataDict.FORCE_KEY,
            AtomicDataDict.TOTAL_ENERGY_KEY,
        ],
        scale_by=forces_std,
        shift_keys=AtomicDataDict.TOTAL_ENERGY_KEY,
        shift_by=energies_mean,
    )

    if config.compile_model:
        core_model = e3nn.util.jit.script(core_model)

    logging.debug(
        f"Outputs are scaled by: {forces_std}, eneriges are shifted by {energies_mean}"
    )

    # Set the trainer
    setattr(core_model, ORIG_CONFIG_KEY, dict(config))
    trainer.model = core_model

    # Train
    trainer.train()

    return


if __name__ == "__main__":
    main()
