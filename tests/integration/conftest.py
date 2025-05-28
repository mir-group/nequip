import pytest
import tempfile
import pathlib
import subprocess
import os
import sys
from omegaconf import OmegaConf, open_dict


def _check_and_print(retcode):
    __tracebackhide__ = True
    if retcode.returncode:
        if retcode.stdout is not None and len(retcode.stdout) > 0:
            print(retcode.stdout.decode("ascii"))
        if retcode.stderr is not None and len(retcode.stderr) > 0:
            print(retcode.stderr.decode("ascii"), file=sys.stderr)
        retcode.check_returncode()


def _training_session(conffile, training_module, model_dtype, extra_train_from_save):
    path_to_this_file = pathlib.Path(__file__)
    config_path = path_to_this_file.parents[0] / conffile
    config = OmegaConf.load(config_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        with tempfile.TemporaryDirectory() as data_tmpdir:
            if "data_source_dir" in config.data:
                config.data.data_source_dir = data_tmpdir
            config.training_module._target_ = training_module
            config.training_module.model.model_dtype = model_dtype
            config.data.val_dataloader.batch_size = 1
            with open_dict(config):
                config["hydra"] = {"run": {"dir": tmpdir}}
                # mitigate nondeterminism for tests
                config.trainer["num_sanity_val_steps"] = 0
                config.trainer["deterministic"] = True
            config = OmegaConf.create(config)
            config_path = tmpdir + "/conf.yaml"
            OmegaConf.save(config=config, f=config_path)

            # == Train model ==
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

            if extra_train_from_save is not None:
                with tempfile.TemporaryDirectory() as new_tmpdir:
                    new_config = config.copy()
                    with open_dict(config):
                        config["hydra"] = {"run": {"dir": new_tmpdir}}
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


@pytest.fixture(
    scope="session",
    params=["minimal_aspirin.yaml", "minimal_toy_emt.yaml"],
)
def conffile(request):
    return request.param


@pytest.fixture(
    scope="session",
    # to save time, we only check the EMA ones for now
    params=[
        # "nequip.train.NequIPLightningModule",
        "nequip.train.EMALightningModule",
        # "nequip.train.ConFIGLightningModule",
        "nequip.train.EMAConFIGLightningModule",
    ],
)
def training_module(request):
    return request.param


@pytest.fixture(scope="session", params=[None, "checkpoint", "package"])
def extra_train_from_save(request):
    """
    Whether the checkpoints for the tests come from a fresh model, `ModelFromCheckpoint`, or `ModelFromPackage`.
    """
    return request.param


@pytest.fixture(scope="session")
def fake_model_training_session(
    conffile, training_module, model_dtype, extra_train_from_save
):
    session = _training_session(
        conffile, training_module, model_dtype, extra_train_from_save
    )
    config, tmpdir, env = next(session)
    yield config, tmpdir, env, model_dtype
    del session
