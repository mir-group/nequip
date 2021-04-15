""" Train a network."""
import logging
import argparse
import yaml

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

import torch

import e3nn.util.jit

from nequip.utils import Config, dataset_from_config, Output
from nequip.data import AtomicDataDict
from nequip.nn import RescaleOutput
from nequip.utils.test import assert_AtomicData_equivariant, set_irreps_debug


def main(args=None):
    parser = argparse.ArgumentParser(description="Train a NequIP model.")
    parser.add_argument("config", help="configuration file")
    parser.add_argument(
        "--equivariance-test",
        help="test the model's equivariance before training",
        action="store_true",
    )
    parser.add_argument(
        "--debug-mode",
        help="enable debug mode (sometimes can give more helpful error messages)",
        action="store_true",
    )
    args = parser.parse_args(args=args)

    config = Config.from_file(
        args.config,
        defaults=dict(
            wandb=False,
            compile_model=False,
            wandb_project="NequIP",
            model_builder="nequip.models.ForceModel",
            force_training=True,
        ),
    )

    torch.set_default_dtype(torch.float32)
    output = Output.from_config(config)
    config.update(output.updated_dict())

    if args.debug_mode:
        set_irreps_debug(enabled=True)

    # Make the trainer
    if config.wandb:
        import wandb  # noqa: F401
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
    stats_fields = [
        AtomicDataDict.TOTAL_ENERGY_KEY,
        AtomicDataDict.ATOMIC_NUMBERS_KEY,
    ]
    stats_modes = ["mean_std", "count"]
    if config.force_training:
        stats_fields.append(AtomicDataDict.FORCE_KEY)
        stats_modes.append("rms")
    stats = trainer.dataset_train.statistics(
        fields=stats_fields,
        modes=stats_modes,
    )
    (
        (energies_mean, energies_scale),
        (allowed_species, Z_count),
    ) = stats[:2]
    if config.force_training:
        # Scale by the force std instead
        energies_scale = stats[2][0]
    del stats_modes
    del stats_fields

    RESCALE_THRESHOLD = 1e-6
    if energies_scale < RESCALE_THRESHOLD:
        raise ValueError(f"RMS of forces in this dataset was very low: {forces_std}")
        # TODO: offer option to disable rescaling?

    config.update(dict(allowed_species=allowed_species))

    # Build a model
    model_builder = config.model_builder
    model_builder = yaml.load(f"!!python/name:{model_builder}", Loader=yaml.Loader)
    assert callable(model_builder), f"Model builder {model_builder} isn't callable"
    core_model = model_builder(**dict(config))

    logging.info("Successfully built the network...")

    final_model = RescaleOutput(
        model=core_model,
        scale_keys=[AtomicDataDict.TOTAL_ENERGY_KEY]
        + (
            [AtomicDataDict.FORCE_KEY]
            if AtomicDataDict.FORCE_KEY in core_model.irreps_out
            else []
        ),
        scale_by=energies_scale,
        shift_keys=AtomicDataDict.TOTAL_ENERGY_KEY,
        shift_by=energies_mean,
    )

    if config.compile_model:
        final_model = e3nn.util.jit.script(final_model)

    logging.debug(
        f"Outputs are scaled by: {energies_scale}, eneriges are shifted by {energies_mean}"
    )

    # Record final config
    with open(output.generate_file("config_final.yaml"), "w+") as fp:
        yaml.dump(dict(config), fp)

    # Equivar test
    if args.equivariance_test:
        equivar_err = assert_AtomicData_equivariant(final_model, dataset.get(0))
        errstr = "\n".join(
            f"    parity_k={parity_k.item()}, did_translate={did_trans} -> max componentwise error={err.item()}"
            for (parity_k, did_trans), err in equivar_err.items()
        )
        del equivar_err
        logging.info(f"Equivariance test passed; equivariance errors:\n{errstr}")
        del errstr

    # Set the trainer
    trainer.model = final_model

    # Train
    trainer.train()

    return


if __name__ == "__main__":
    main()
