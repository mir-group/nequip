# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import copy
import tempfile
import pathlib
import subprocess
import os
import sys
import torch

from omegaconf import OmegaConf, open_dict
from nequip.data import AtomicDataDict


def _check_and_print(retcode, encoding="ascii"):
    """Helper function to check subprocess return code and print output on failure."""
    __tracebackhide__ = True
    if retcode.returncode:
        if retcode.stdout is not None and len(retcode.stdout) > 0:
            print(retcode.stdout.decode(encoding, errors="replace"))
        if retcode.stderr is not None and len(retcode.stderr) > 0:
            print(retcode.stderr.decode(encoding, errors="replace"), file=sys.stderr)
        retcode.check_returncode()


def compare_output_and_gradients(
    modelA, modelB, model_test_data, tol, compare_outputs=None
):
    """Compare model outputs and parameter gradients for consistency."""
    # default fields
    if compare_outputs is None:
        compare_outputs = [
            AtomicDataDict.PER_ATOM_ENERGY_KEY,
            AtomicDataDict.TOTAL_ENERGY_KEY,
            AtomicDataDict.FORCE_KEY,
            AtomicDataDict.VIRIAL_KEY,
        ]

    A_out = modelA(model_test_data.copy())
    B_out = modelB(model_test_data.copy())
    for key in compare_outputs:
        if key in A_out and key in B_out:
            torch.testing.assert_close(A_out[key], B_out[key], atol=tol, rtol=tol)

    # test backwards pass if there are trainable weights
    if any([p.requires_grad for p in modelB.parameters()]):
        B_loss = B_out[AtomicDataDict.TOTAL_ENERGY_KEY].square().sum()
        B_loss.backward()

        A_loss = A_out[AtomicDataDict.TOTAL_ENERGY_KEY].square().sum()
        A_loss.backward()
        compile_params = dict(modelB.named_parameters())
        for k, v in modelA.named_parameters():
            err = torch.max(torch.abs(v.grad - compile_params[k].grad))
            torch.testing.assert_close(
                v.grad,
                compile_params[k].grad,
                atol=tol,
                rtol=tol,
                msg=f"failed for {k}, with MaxAbsErr of {err:.6f}",
            )


def _training_session(
    conffile,
    model_dtype,
    extra_train_from_save="fresh",
    model_config=None,
    training_module_override_dict=None,
):
    """
    Create a training session using config files with optional model injection.

    Unified function for both integration and unit test training sessions.

    Args:
        conffile: Name of config file (e.g., "minimal_aspirin.yaml")
        model_dtype: Model dtype string (e.g., "float32", "float64")
        extra_train_from_save: One of "fresh", "checkpoint", or "package" for additional training
        model_config: Optional model config dict to inject (for unit tests)
        training_module_override_dict: Optional dict with training_module override, including optimizer (e.g. for ScheduleFreeLightningModule)
    Yields:
        tuple: (config, tmpdir, env) - training config, temp directory, and env vars
    """
    if extra_train_from_save not in ("fresh", "checkpoint", "package"):
        raise ValueError(
            "extra_train_from_save must be one of 'fresh', 'checkpoint', or 'package'. "
            f"Got: {extra_train_from_save}"
        )

    # find the config file in the same directory as this utils file
    current_file = pathlib.Path(__file__)
    config_path = current_file.parent / conffile
    assert config_path.exists(), (
        f"Could not find config file: {conffile} in path: {config_path}"
    )

    config = OmegaConf.load(config_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        with tempfile.TemporaryDirectory() as data_tmpdir:
            if "data_source_dir" in config.data:
                config.data.data_source_dir = data_tmpdir

            # configure training module and model
            if training_module_override_dict is not None:
                # integration test case: use provided training module
                if training_module_override_dict:
                    with open_dict(config):
                        config.training_module.update(training_module_override_dict)
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
                config.trainer["deterministic"] = "warn"
                # ^ warn so that we can use nondeterministic kernels, e.g. from OpenEquivariance

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

            # handle extra training from save
            if extra_train_from_save == "fresh":
                yield config, tmpdir, env
            else:
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
                                f"{tmpdir}/last.ckpt",
                                package_path,
                            ],
                            cwd=new_tmpdir,
                            env=env,
                        )
                        _check_and_print(retcode)
                        assert pathlib.Path(package_path).is_file(), (
                            "`nequip-package` didn't create file"
                        )
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
