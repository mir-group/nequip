""" Train a network."""
import logging
import argparse

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

from os.path import isdir
from pathlib import Path

import torch

import e3nn
import e3nn.util.jit

from nequip.model import model_from_config
from nequip.utils import Config
from nequip.data import dataset_from_config
from nequip.utils.test import assert_AtomicData_equivariant, set_irreps_debug
from nequip.utils import load_file, dtype_from_name
from nequip.scripts.logger import set_up_script_logger

default_config = dict(
    root="./",
    run_name="NequIP",
    wandb=False,
    wandb_project="NequIP",
    compile_model=False,
    model_builders=[
        "EnergyModel",
        "PerSpeciesRescale",
        "ForceOutput",
        "RescaleEnergyEtc",
    ],
    dataset_statistics_stride=1,
    default_dtype="float32",
    allow_tf32=False,  # TODO: until we understand equivar issues
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    append=False,
    _jit_bailout_depth=2,  # avoid 20 iters of pain, see https://github.com/pytorch/pytorch/issues/52286
)


def main(args=None, running_as_script: bool = True):

    config = parse_command_line(args)

    if running_as_script:
        set_up_script_logger(config.get("log", None), config.verbose)

    found_restart_file = isdir(f"{config.root}/{config.run_name}")
    if found_restart_file and not config.append:
        raise RuntimeError(
            f"Training instance exists at {config.root}/{config.run_name}; "
            "either set append to True or use a different root or runname"
        )

    # for fresh new train
    if not found_restart_file:
        trainer = fresh_start(config)
    else:
        trainer = restart(config)

    # Train
    trainer.save()
    trainer.train()

    return


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
    parser.add_argument(
        "--log",
        help="log file to store all the screen logging",
        type=Path,
        default=None,
    )
    args = parser.parse_args(args=args)

    config = Config.from_file(args.config, defaults=default_config)
    for flag in ("model_debug_mode", "equivariance_test", "grad_anomaly_mode"):
        config[flag] = getattr(args, flag) or config[flag]

    return config


def _set_global_options(config):
    """Configure global options of libraries like `torch` and `e3nn` based on `config`."""
    # Set TF32 support
    # See https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available():
        if torch.torch.backends.cuda.matmul.allow_tf32 and not config.allow_tf32:
            # it is enabled, and we dont want it to, so disable:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    # For avoiding 20 steps of painfully slow JIT recompilation
    # See https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_bailout_depth(config["_jit_bailout_depth"])

    if config.model_debug_mode:
        set_irreps_debug(enabled=True)
    torch.set_default_dtype(dtype_from_name(config.default_dtype))
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

    # what is this
    # to update wandb data?
    config.update(trainer.params)

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

    # = Train/test split =
    trainer.set_dataset(dataset, validation_dataset)

    # = Build model =
    final_model = model_from_config(
        config=config, initialize=True, dataset=trainer.dataset_train
    )

    logging.info("Successfully built the network...")

    if config.compile_model:
        final_model = e3nn.util.jit.script(final_model)
        logging.info("Successfully compiled model...")

    # Equivar test
    if config.equivariance_test:
        from e3nn.util.test import format_equivariance_error

        equivar_err = assert_AtomicData_equivariant(final_model, dataset[0])
        errstr = format_equivariance_error(equivar_err)
        del equivar_err
        logging.info(f"Equivariance test passed; equivariance errors:\n{errstr}")
        del errstr

    # Set the trainer
    trainer.model = final_model

    # Store any updated config information in the trainer
    trainer.update_kwargs(config)

    return trainer


def restart(config):

    # load the dictionary
    restart_file = f"{config.root}/{config.run_name}/trainer.pth"
    dictionary = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename=restart_file,
        enforced_format="torch",
    )

    # compare dictionary to config and update stop condition related arguments
    for k in config.keys():
        if config[k] != dictionary.get(k, ""):
            if k == "max_epochs":
                dictionary[k] = config[k]
                logging.info(f'Update "{k}" to {dictionary[k]}')
            elif k.startswith("early_stop"):
                dictionary[k] = config[k]
                logging.info(f'Update "{k}" to {dictionary[k]}')
            elif isinstance(config[k], type(dictionary.get(k, ""))):
                raise ValueError(
                    f'Key "{k}" is different in config and the result trainer.pth file. Please double check'
                )

    # recursive loop, if same type but different value
    # raise error

    config = Config(dictionary, exclude_keys=["state_dict", "progress"])

    # dtype, etc.
    _set_global_options(config)

    if config.wandb:
        from nequip.train.trainer_wandb import TrainerWandB
        from nequip.utils.wandb import resume

        resume(config)
        trainer = TrainerWandB.from_dict(dictionary)
    else:
        from nequip.train.trainer import Trainer

        trainer = Trainer.from_dict(dictionary)

    # = Load the dataset =
    dataset = dataset_from_config(config, prefix="dataset")
    logging.info(f"Successfully re-loaded the data set of type {dataset}...")
    try:
        validation_dataset = dataset_from_config(config, prefix="validation_dataset")
        logging.info(
            f"Successfully re-loaded the validation data set of type {validation_dataset}..."
        )
    except KeyError:
        # It couldn't be found
        validation_dataset = None
    trainer.set_dataset(dataset, validation_dataset)

    return trainer


if __name__ == "__main__":
    main(running_as_script=True)
