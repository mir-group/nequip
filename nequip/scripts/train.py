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

default_config = dict(
    wandb=False,
    compile_model=False,
    wandb_project="NequIP",
    model_builder="nequip.models.ForceModel",
    dataset_statistics_stride=1,
    default_dtype="float32",
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
)


def main(args=None):
    fresh_start(parse_command_line(args))


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(description="Train a NequIP model.")
    parser.add_argument("config", help="configuration file")
    parser.add_argument(
        "--equivariance-test",
        help="test the model's equivariance before training",
        action="store_true",
    )
    parser.add_argument(
        "--model-debug-mode",
        help="enable model debug mode, which can sometimes give much more useful error messages at the cost of some speed. Do not use for production training!",
        action="store_true",
    )
    args = parser.parse_args(args=args)

    config = Config.from_file(args.config, defaults=default_config)
    config.model_debug_mode = args.model_debug_mode or config.model_debug_mode
    config.equivariance_test = args.equivariance_test or config.equivariance_test

    return config


def fresh_start(config):
    if config.model_debug_mode:
        set_irreps_debug(enabled=True)
    torch.set_default_dtype(
        {"float32": torch.float32, "float64": torch.float64}[config.default_dtype]
    )
    output = Output.from_config(config)
    config.update(output.updated_dict())

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

    # Determine training type
    train_on = config.loss_coeffs
    train_on = [train_on] if isinstance(train_on, str) else train_on
    train_on = set(train_on)
    if not train_on.issubset({"forces", "total_energy"}):
        raise NotImplementedError(
            f"Training on fields `{train_on}` besides forces and total energy not supported in the out-of-the-box training script yet; please use your own training script based on train.py."
        )
    force_training = "forces" in train_on
    logging.debug(f"Force training mode: {force_training}")
    del train_on

    # Get statistics of training dataset
    stats_fields = [
        AtomicDataDict.TOTAL_ENERGY_KEY,
        AtomicDataDict.ATOMIC_NUMBERS_KEY,
    ]
    stats_modes = ["mean_std", "count"]
    if force_training:
        stats_fields.append(AtomicDataDict.FORCE_KEY)
        stats_modes.append("rms")
    stats = trainer.dataset_train.statistics(
        fields=stats_fields, modes=stats_modes, stride=config.dataset_statistics_stride
    )
    (
        (energies_mean, energies_scale),
        (allowed_species, Z_count),
    ) = stats[:2]
    if force_training:
        # Scale by the force std instead
        energies_scale = stats[2][0]
    del stats_modes
    del stats_fields

    RESCALE_THRESHOLD = 1e-6
    if energies_scale < RESCALE_THRESHOLD:
        # TODO: move this after merge
        raise ValueError(
            f"RMS of forces/stdev of energies in this dataset was very low: {energies_scale}"
        )
        # TODO: offer option to disable rescaling?

    config.update(dict(allowed_species=allowed_species))

    # Build a model
    if not callable(config.model_builder):
        model_builder = yaml.load(
            f"!!python/name:{config.model_builder}", Loader=yaml.Loader
        )
    assert callable(model_builder), f"Model builder {model_builder} isn't callable"
    core_model = model_builder(**dict(config))

    global_shift = config.get("global_rescale_shift", energies_mean)
    global_scale = config.get("global_rescale_scale", energies_scale)

    final_model = RescaleOutput(
        model=core_model,
        scale_keys=[AtomicDataDict.TOTAL_ENERGY_KEY]
        + (
            [AtomicDataDict.FORCE_KEY]
            if AtomicDataDict.FORCE_KEY in core_model.irreps_out
            else []
        ),
        scale_by=global_scale,
        shift_keys=AtomicDataDict.TOTAL_ENERGY_KEY,
        shift_by=global_shift,
        trainable_global_rescale_shift=config.get(
            "trainable_global_rescale_shift", False
        ),
        trainable_global_rescale_scale=config.get(
            "trainable_global_rescale_scale", False
        ),
    )

    logging.info("Successfully built the network...")

    if config.compile_model:
        final_model = e3nn.util.jit.script(final_model)
        logging.info("Successfully compiled model...")

    logging.debug(
        f"Initially outputs are scaled by: {energies_scale}, eneriges are shifted by {energies_mean}. Scaling factors derived from statistics of {'forces' if force_training else 'energies'} in the dataset."
    )

    # Record final config
    with open(output.generate_file("config_final.yaml"), "w+") as fp:
        yaml.dump(dict(config), fp)

    # Equivar test
    if config.equivariance_test:
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
