""" Train a network."""
from typing import Union, Callable
import logging
import argparse
import yaml

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

import torch

import e3nn
import e3nn.util.jit

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict
from nequip.nn import RescaleOutput
from nequip.utils.test import assert_AtomicData_equivariant, set_irreps_debug

default_config = dict(
    requeue=False,
    wandb=False,
    wandb_project="NequIP",
    wandb_resume=False,
    compile_model=False,
    model_builder="nequip.models.ForceModel",
    model_initializers=[],
    dataset_statistics_stride=1,
    default_dtype="float32",
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
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
    parser.add_argument(
        "--grad-anomaly-mode",
        help="enable PyTorch autograd anomaly mode to debug NaN gradients. Do not use for production training!",
        action="store_true",
    )
    args = parser.parse_args(args=args)

    config = Config.from_file(args.config, defaults=default_config)
    for flag in ("model_debug_mode", "equivariance_test", "grad_anomaly_mode"):
        config[flag] = getattr(args, flag) or config[flag]

    return config


def _load_callable(obj: Union[str, Callable]) -> Callable:
    if callable(obj):
        pass
    elif isinstance(obj, str):
        obj = yaml.load(f"!!python/name:{obj}", Loader=yaml.Loader)
    else:
        raise TypeError
    assert callable(obj), f"{obj} isn't callable"
    return obj


def fresh_start(config):
    # = Set global state =
    if config.model_debug_mode:
        set_irreps_debug(enabled=True)
    torch.set_default_dtype(
        {"float32": torch.float32, "float64": torch.float64}[config.default_dtype]
    )
    if config.grad_anomaly_mode:
        torch.autograd.set_detect_anomaly(True)

    e3nn.set_optimization_defaults(**config.get("e3nn_optimization_defaults", {}))

    # = Make the trainer =
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

    output = trainer.output
    config.update(output.updated_dict())

    # = Load the dataset =
    dataset = dataset_from_config(config)
    logging.info(f"Successfully loaded the data set of type {dataset}...")

    # = Train/test split =
    trainer.set_dataset(dataset)

    # = Determine training type =
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

    # = Get statistics of training dataset =
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
        (energies_mean, energies_std),
        (allowed_species, Z_count),
    ) = stats[:2]
    if force_training:
        # Scale by the force std instead
        force_rms = stats[2][0]
    del stats_modes
    del stats_fields

    config.update(dict(allowed_species=allowed_species))

    # = Build a model =
    model_builder = _load_callable(config.model_builder)
    core_model = model_builder(**dict(config))

    # = Reinit if wanted =
    with torch.no_grad():
        for initer in config.model_initializers:
            initer = _load_callable(initer)
            core_model.apply(initer)

    # = Determine shifts, scales =
    # This is a bit awkward, but necessary for there to be a value
    # in the config that signals "use dataset"
    global_shift = config.get("global_rescale_shift", "dataset_energy_mean")
    if global_shift == "dataset_energy_mean":
        global_shift = energies_mean
    elif (
        global_shift is None
        or isinstance(global_shift, float)
        or isinstance(global_shift, torch.Tensor)
    ):
        # valid values
        pass
    else:
        raise ValueError(f"Invalid global shift `{global_shift}`")

    global_scale = config.get(
        "global_rescale_scale", force_rms if force_training else energies_std
    )
    if global_scale == "dataset_energy_std":
        global_scale = energies_std
    elif global_scale == "dataset_force_rms":
        if not force_training:
            raise ValueError(
                "Cannot have global_scale = 'dataset_force_rms' without force training"
            )
        global_scale = force_rms
    elif (
        global_scale is None
        or isinstance(global_scale, float)
        or isinstance(global_scale, torch.Tensor)
    ):
        # valid values
        pass
    else:
        raise ValueError(f"Invalid global scale `{global_scale}`")

    RESCALE_THRESHOLD = 1e-6
    if isinstance(global_scale, float) and global_scale < RESCALE_THRESHOLD:
        raise ValueError(
            f"Global energy scaling was very low: {global_scale}. If dataset values were used, does the dataset contain insufficient variation? Maybe try disabling global scaling with global_scale=None."
        )
        # TODO: offer option to disable rescaling?

    logging.debug(
        f"Initially outputs are scaled by: {global_scale}, eneriges are shifted by {global_shift}."
    )

    # == Build the model ==
    final_model = RescaleOutput(
        model=core_model,
        scale_keys=[AtomicDataDict.TOTAL_ENERGY_KEY, AtomicDataDict.PER_ATOM_ENERGY_KEY]
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
