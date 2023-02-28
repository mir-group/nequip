""" Train a network."""
import logging
import argparse
import warnings

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

from os.path import isdir
from pathlib import Path

import torch

from nequip.model import model_from_config
from nequip.utils import Config
from nequip.data import dataset_from_config
from nequip.utils import load_file
from nequip.utils.config import _GLOBAL_ALL_ASKED_FOR_KEYS
from nequip.utils.test import assert_AtomicData_equivariant
from nequip.utils.versions import check_code_version
from nequip.utils._global_options import _set_global_options
from nequip.scripts._logger import set_up_script_logger

default_config = dict(
    root="./",
    tensorboard=False,
    wandb=False,
    model_builders=[
        "SimpleIrrepsConfig",
        "EnergyModel",
        "PerSpeciesRescale",
        "StressForceOutput",
        "RescaleEnergyEtc",
    ],
    dataset_statistics_stride=1,
    default_dtype="float64",
    model_dtype="float32",
    allow_tf32=False,
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    gpu_oom_offload=False,
    append=False,
    warn_unused=False,
    _jit_bailout_depth=2,  # avoid 20 iters of pain, see https://github.com/pytorch/pytorch/issues/52286
    # Quote from eelison in PyTorch slack:
    # https://pytorch.slack.com/archives/CDZD1FANA/p1644259272007529?thread_ts=1644064449.039479&cid=CDZD1FANA
    # > Right now the default behavior is to specialize twice on static shapes and then on dynamic shapes.
    # > To reduce warmup time you can do something like setFusionStrartegy({{FusionBehavior::DYNAMIC, 3}})
    # > ... Although we would wouldn't really expect to recompile a dynamic shape fusion in a model,
    # > provided broadcasting patterns remain fixed
    # We default to DYNAMIC alone because the number of edges is always dynamic,
    # even if the number of atoms is fixed:
    _jit_fusion_strategy=[("DYNAMIC", 3)],
    # Due to what appear to be ongoing bugs with nvFuser, we default to NNC (fuser1) for now:
    _jit_fuser="fuser1",
)
# All default_config keys are valid / requested
_GLOBAL_ALL_ASKED_FOR_KEYS.update(default_config.keys())


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
    if config.get("gpu_oom_offload", False):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available; --gpu-oom-offload doesn't make sense."
            )
        warnings.warn(
            "! GPU OOM Offloading is ON:\n"
            "This is meant for training models that would be impossible otherwise due to OOM.\n"
            "Note that this comes at a speed cost and SHOULD NOT be used if your training fits in GPU memory without it.\n"
            "Please also consider whether a smaller model is a more appropriate solution!\n"
            "Also, a warning from PyTorch: 'If you overuse pinned memory, it can cause serious problems when running low on RAM!'"
        )
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            trainer.train()
    else:
        trainer.train()

    return


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(
        description="Train (or restart training of) a NequIP model."
    )
    parser.add_argument(
        "config", help="YAML file configuring the model, dataset, and other options"
    )
    parser.add_argument(
        "--equivariance-test",
        help="test the model's equivariance before training on n (default 1) random frames from the dataset",
        const=1,
        type=int,
        nargs="?",
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
        "--gpu-oom-offload",
        help="Use `torch.autograd.graph.save_on_cpu` to offload intermediate tensors to CPU (host) memory in order to train models that would be impossible otherwise due to OOM. Note that this comes as at a speed cost and SHOULD NOT be used if your training fits in GPU memory without it. Please also consider whether a smaller model is a more appropriate solution.",
        action="store_true",
    )
    parser.add_argument(
        "--log",
        help="log file to store all the screen logging",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--warn-unused",
        help="Warn instead of error when the config contains unused keys",
        action="store_true",
    )
    args = parser.parse_args(args=args)

    config = Config.from_file(args.config, defaults=default_config)
    for flag in (
        "model_debug_mode",
        "equivariance_test",
        "grad_anomaly_mode",
        "warn_unused",
        "gpu_oom_offload",
    ):
        config[flag] = getattr(args, flag) or config[flag]

    return config


def fresh_start(config):
    # we use add_to_config cause it's a fresh start and need to record it
    check_code_version(config, add_to_config=True)
    _set_global_options(config)
    if config["default_dtype"] != "float64":
        warnings.warn(
            f"default_dtype={config['default_dtype']} but we strongly recommend float64"
        )

    # = Make the trainer =
    if config.wandb:

        import wandb  # noqa: F401
        from nequip.train.trainer_wandb import TrainerWandB as Trainer

        # download parameters from wandb in case of sweeping
        from nequip.utils.wandb import init_n_update

        config = init_n_update(config)

    elif config.tensorboard:
        from nequip.train.trainer_tensorboard import TrainerTensorBoard as Trainer
    else:
        from nequip.train.trainer import Trainer

    trainer = Trainer(model=None, **Config.as_dict(config))

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

    # Equivar test
    if config.equivariance_test > 0:
        n_train: int = len(trainer.dataset_train)
        assert config.equivariance_test <= n_train
        final_model.eval()
        indexes = torch.randperm(n_train)[: config.equivariance_test]
        errstr = assert_AtomicData_equivariant(
            final_model, [trainer.dataset_train[i] for i in indexes]
        )
        final_model.train()
        logging.info(
            "Equivariance test passed; equivariance errors:\n"
            "   Errors are in real units, where relevant.\n"
            "   Please note that the large scale of the typical\n"
            "   shifts to the (atomic) energy can cause\n"
            "   catastrophic cancellation and give incorrectly\n"
            "   the equivariance error as zero for those fields.\n"
            f"{errstr}"
        )
        del errstr, indexes, n_train

    # Set the trainer
    trainer.model = final_model

    # Store any updated config information in the trainer
    trainer.update_kwargs(config)

    # Only run the unused check as a callback after the trainer has
    # initialized everything (metrics, early stopping, etc.)
    def _unused_check():
        unused = config._unused_keys()
        if len(unused) > 0:
            message = f"The following keys in the config file were not used, did you make a typo?: {', '.join(unused)}. (If this sounds wrong, please file an issue: the detection of unused keys is in beta. You can turn this error into a warning with `--warn-unused`.)"
            if config.warn_unused:
                warnings.warn(message)
            else:
                raise KeyError(message)

    trainer._post_init_callback = _unused_check

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

    # note, "trainer.pth"/dictionary also store code versions,
    # which will not be stored in config and thus not checked here
    check_code_version(config)

    # recursive loop, if same type but different value
    # raise error

    config = Config(dictionary, exclude_keys=["state_dict", "progress"])

    # dtype, etc.
    _set_global_options(config)

    # note, the from_dict method will check whether the code version
    # in trainer.pth is consistent and issue warnings
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
