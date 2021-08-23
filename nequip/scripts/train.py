""" Train a network."""
from typing import Union, Callable
import logging
import argparse
from torch._C import Value
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
    allow_tf32=False,  # TODO: until we understand equivar issues
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


def _set_global_options(config):
    """Configure global options of libraries like `torch` and `e3nn` based on `config`."""
    # Set TF32 support
    # See https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available():
        if torch.torch.backends.cuda.matmul.allow_tf32 and not config.allow_tf32:
            # it is enabled, and we dont want it to, so disable:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    if config.model_debug_mode:
        set_irreps_debug(enabled=True)
    torch.set_default_dtype(
        {"float32": torch.float32, "float64": torch.float64}[config.default_dtype]
    )
    if config.grad_anomaly_mode:
        torch.autograd.set_detect_anomaly(True)

    e3nn.set_optimization_defaults(**config.get("e3nn_optimization_defaults", {}))


def fresh_start(config):
    _set_global_options(config)

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
    dataset = dataset_from_config(config, prefix="dataset")
    logging.info(f"Successfully loaded the data set of type {dataset}...")
    try:
        validation_dataset = dataset_from_config(config, prefix="validation_dataset")
        logging.info(
            f"Successfully loaded the validation data set of type {validation_dataset}..."
        )
    except KeyError:
        # It couldn't be found
        validation_dataset = None

    # For the model building:
    config["num_types"] = dataset.type_mapper.num_types
    config["type_names"] = dataset.type_mapper.type_names

    # = Train/test split =
    trainer.set_dataset(dataset, validation_dataset)

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

    # = Determine energy rescale type =
    global_shift = config.get("global_rescale_shift", "dataset_energy_mean")
    global_scale = config.get(
        "global_rescale_scale",
        "dataset_force_rms" if force_training else "dataset_energy_std",
    )

    def get_per_species(key, default):
        return config.get(
            f"PerSpeciesScaleShift_{key}",
            config.get(f"per_species_scale_shift_{key}", default),
        )

    def pop_per_species(key, default):
        return config.pop(
            f"PerSpeciesScaleShift_{key}",
            config.pop(f"per_species_scale_shift_{key}", default),
        )

    per_species_scale_shift = get_per_species("enable", False)
    if global_shift is not None and per_species_scale_shift:
        raise ValueError("One can only enable either global shift or per_species shift")
    logging.debug(f"Enable per species scale shift: {per_species_scale_shift}")
    logging.debug(f"Enable global scale shift: {per_species_scale_shift}")

    # = Get statistics of training dataset =
    if force_training:
        ((force_rms,),) = trainer.dataset_train.statistics(
            fields=[AtomicDataDict.FORCE_KEY],
            modes=["rms"],
            stride=config.dataset_statistics_stride,
        )
    if global_scale == "dataset_energy_std" or global_shift == "dataset_energy_mean":
        ((energies_mean, energies_std),) = trainer.dataset_train.statistics(
            fields=[AtomicDataDict.TOTAL_ENERGY_KEY],
            modes=["mean_std"],
            stride=config.dataset_statistics_stride,
        )

    # = Determine shifts, scales =
    # This is a bit awkward, but necessary for there to be a value
    # in the config that signals "use dataset"

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

    if per_species_scale_shift:
        scales = pop_per_species("scales", None)
        shifts = pop_per_species("shifts", None)
        if scales == "dataset_energy_std" or shifts == "dataset_energy_mean":
            (
                (per_species_energies_mean, per_species_energies_std),
            ) = trainer.dataset_train.statistics(
                fields=[AtomicDataDict.TOTAL_ENERGY_KEY],
                modes=["atom_type_mean_std"],
                stride=config.dataset_statistics_stride,
            )

        if scales == "dataset_energy_std":
            scales = per_species_energies_std
            if torch.min(scales) < RESCALE_THRESHOLD:
                raise ValueError(
                    f"Atomic energy scaling was very low: {torch.min(scales)}. "
                    "If dataset values were used, does the dataset contain insufficient variation? Maybe try disabling global scaling with global_scale=None."
                )
        elif (
            scales is None
            or isinstance(scales, float)
            or isinstance(scales, torch.Tensor)
        ):
            pass
        else:
            raise ValueError(f"Scales has to be number or but {scales}")

        if global_scale is not None and scales is not None:
            scales = scales / global_scale
        config["PerSpeciesScaleShift_scales"] = scales

        if shifts == "dataset_energy_mean":
            shifts = per_species_energies_mean
        elif (
            shifts is None
            or isinstance(shifts, float)
            or isinstance(shifts, torch.Tensor)
        ):
            pass
        else:
            raise ValueError(f"Shifts has to be number but {shifts}")
        if global_scale is not None and shifts is not None:
            shifts = shifts / global_scale
        config["PerSpeciesScaleShift_shifts"] = shifts
        logging.debug(
            f"Initially per atom outputs are scaled by: {scales}, eneriges are shifted by {shifts}."
        )
    # raise RuntimeError("hello")

    # = Build a model =
    model_builder = _load_callable(config.model_builder)
    core_model = model_builder(**dict(config))

    # = Reinit if wanted =
    with torch.no_grad():
        for initer in config.model_initializers:
            initer = _load_callable(initer)
            core_model.apply(initer)

    # == Build the model ==
    final_model = RescaleOutput(
        model=core_model,
        scale_keys=[AtomicDataDict.TOTAL_ENERGY_KEY]
        + (
            [AtomicDataDict.FORCE_KEY]
            if AtomicDataDict.FORCE_KEY in core_model.irreps_out
            else []
        )
        + (
            [AtomicDataDict.PER_ATOM_ENERGY_KEY]
            if AtomicDataDict.PER_ATOM_ENERGY_KEY in core_model.irreps_out
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
        from e3nn.util.test import format_equivariance_error

        equivar_err = assert_AtomicData_equivariant(final_model, dataset.get(0))
        errstr = format_equivariance_error(equivar_err)
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
