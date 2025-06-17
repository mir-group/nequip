# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import copy
import tempfile
import pathlib
import subprocess
import os
import sys

from omegaconf import OmegaConf, open_dict


def _check_and_print(retcode):
    """Helper function to check subprocess return code and print output on failure."""
    __tracebackhide__ = True
    if retcode.returncode:
        if retcode.stdout is not None and len(retcode.stdout) > 0:
            print(retcode.stdout.decode("ascii", errors="replace"))
        if retcode.stderr is not None and len(retcode.stderr) > 0:
            print(retcode.stderr.decode("ascii", errors="replace"), file=sys.stderr)
        retcode.check_returncode()


def _training_session(
    conffile,
    model_dtype,
    extra_train_from_save=None,
    model_config=None,
    training_module=None,
):
    """
    Create a training session using config files with optional model injection.

    Unified function for both integration and unit test training sessions.

    Args:
        conffile: Name of config file (e.g., "minimal_aspirin.yaml")
        model_dtype: Model dtype string (e.g., "float32", "float64")
        extra_train_from_save: Optional, None/"checkpoint"/"package" for additional training
        model_config: Optional model config dict to inject (for unit tests)
        training_module: Optional training module target (for integration tests)

    Yields:
        tuple: (config, tmpdir, env) - training config, temp directory, and env vars
    """
    # find the config file in the same directory as this utils file
    current_file = pathlib.Path(__file__)
    config_path = current_file.parent / conffile
    assert (
        config_path.exists()
    ), f"Could not find config file: {conffile} in path: {config_path}"

    config = OmegaConf.load(config_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        with tempfile.TemporaryDirectory() as data_tmpdir:
            if "data_source_dir" in config.data:
                config.data.data_source_dir = data_tmpdir

            # configure training module and model
            if training_module is not None:
                # integration test case: use provided training module
                config.training_module._target_ = training_module
                config.training_module.model.model_dtype = model_dtype
            else:
                # unit test case: default training module
                config.training_module._target_ = "nequip.train.EMALightningModule"

            if model_config is not None:
                # unit test case: replace model entirely
                config.training_module.model = copy.deepcopy(model_config)
                config.training_module.model.model_dtype = model_dtype

            # set training parameters
            config.data.val_dataloader.batch_size = 1
            if model_config is not None:
                # unit tests: minimal training
                config.trainer.max_epochs = 2

            with open_dict(config):
                config["hydra"] = {"run": {"dir": tmpdir}}
                # mitigate nondeterminism for tests
                config.trainer["num_sanity_val_steps"] = 0
                config.trainer["deterministic"] = True

            config = OmegaConf.create(config)
            config_path = tmpdir + "/conf.yaml"
            OmegaConf.save(config=config, f=config_path)

            # == train model ==
            env = dict(os.environ)
            env.update({"TEST_VAL_TWO": "2"})
            retcode = subprocess.run(
                ["nequip-train", "-cn", "conf"],
                cwd=tmpdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _check_and_print(retcode)

            yield config, tmpdir, env

            # handle extra training from save (for integration tests)
            if extra_train_from_save is not None:
                with tempfile.TemporaryDirectory() as new_tmpdir:
                    new_config = config.copy()
                    with open_dict(new_config):
                        new_config["hydra"] = {"run": {"dir": new_tmpdir}}
                    if extra_train_from_save == "checkpoint":
                        with open_dict(new_config):
                            new_config["training_module"]["model"] = {
                                "_target_": "nequip.model.ModelFromCheckpoint",
                                "checkpoint_path": f"{tmpdir}/last.ckpt",
                            }
                    elif extra_train_from_save == "package":
                        # package model
                        package_path = f"{new_tmpdir}/orig_package_model.nequip.zip"
                        retcode = subprocess.run(
                            [
                                "nequip-package",
                                "build",
                                "--ckpt-path",
                                f"{tmpdir}/last.ckpt",
                                "--output-path",
                                package_path,
                            ],
                            cwd=new_tmpdir,
                            env=env,
                        )
                        _check_and_print(retcode)
                        assert pathlib.Path(
                            package_path
                        ).is_file(), "`nequip-package` didn't create file"
                        # update config
                        with open_dict(new_config):
                            new_config["training_module"]["model"] = {
                                "_target_": "nequip.model.ModelFromPackage",
                                "package_path": package_path,
                            }
                    new_config = OmegaConf.create(new_config)
                    config_path = f"{new_tmpdir}/conf.yaml"
                    OmegaConf.save(config=new_config, f=config_path)
                    retcode = subprocess.run(
                        ["nequip-train", "-cn", "conf"],
                        cwd=new_tmpdir,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    _check_and_print(retcode)
                    yield new_config, new_tmpdir, env
