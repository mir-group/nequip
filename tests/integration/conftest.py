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
            print(retcode.stdout.decode("ascii", errors='replace'))
        if retcode.stderr is not None and len(retcode.stderr) > 0:
            print(retcode.stderr.decode("ascii", errors='replace'), file=sys.stderr)
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


# TODO: Move these imports when this class gets moved to the correct file. 
import hydra, math, torch 

class TrainingInvarianceBaseTest:
    def modify_model_config(self, original_config):
        raise NotImplementedError("Implement in subclass test.")

    def map_location(self):
        return "cpu"

    def load_nequip_module_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.map_location(),
            weights_only=False,
        )
        training_module = hydra.utils.get_class(
            checkpoint["hyper_parameters"]["info_dict"]["training_module"]["_target_"]
        )
        nequip_module = training_module.load_from_checkpoint(checkpoint_path)
        return nequip_module


    def test_batch_invariance(self, fake_model_training_session):
        # This actually tests two features:
        # 1. reproducibility of training with the same data and global seeds (we train in exactly the same way as in fake_model_training_session)
        # 2. invariance of validation metrics with the validation batch sizes

        # NOTE: at the time the tests were written, both minimal configs have val dataloader batch sizes of 5, so we only train again with batch_size=1 here
        config, tmpdir, env, model_dtype = fake_model_training_session
        tol = {"float32": 1e-5, "float64": 1e-8}[model_dtype]

        # == get necessary info from checkpoint ==
        nequip_module = self.load_nequip_module_from_checkpoint(f"{tmpdir}/last.ckpt")
        orig_train_loss = nequip_module.loss.metrics_values_epoch
        batchsize5_val_metrics = nequip_module.val_metrics[0].metrics_values_epoch

        # == train again with validation batch size 1 ==
        with tempfile.TemporaryDirectory() as new_tmpdir:
            new_config = config.copy()
            new_config.data.val_dataloader.batch_size = 1
            with open_dict(new_config):
                new_config["hydra"] = {"run": {"dir": new_tmpdir}}
            new_config = self.modify_model_config(new_config)
            new_config = OmegaConf.create(new_config)
            config_path = new_tmpdir + "/newconf.yaml"
            OmegaConf.save(config=new_config, f=config_path)

            # == Train model ==
            retcode = subprocess.run(
                ["nequip-train", "-cn", "newconf"],
                cwd=new_tmpdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _check_and_print(retcode)
            nequip_module = self.load_nequip_module_from_checkpoint(f"{new_tmpdir}/last.ckpt")

            # == test training loss reproduced ==
            new_train_loss = nequip_module.loss.metrics_values_epoch
            print(orig_train_loss)
            print(new_train_loss)
            assert len(orig_train_loss) == len(new_train_loss)
            assert all(
                [
                    math.isclose(a, b, rel_tol=tol)
                    for a, b in zip(orig_train_loss.values(), new_train_loss.values())
                ]
            )

            # == test val metrics invariance to batch size ==
            batchsize1_val_metrics = nequip_module.val_metrics[0].metrics_values_epoch
            print(batchsize5_val_metrics)
            print(batchsize1_val_metrics)
            assert len(batchsize5_val_metrics) == len(batchsize1_val_metrics)
            assert all(
                [
                    math.isclose(a, b, rel_tol=tol)
                    for a, b in zip(
                        batchsize5_val_metrics.values(), batchsize1_val_metrics.values()
                    )
                ]
            )


    # TODO: will fail if train dataloader has shuffle=True
    def test_restarts(self, fake_model_training_session):
        config, tmpdir, env, model_dtype = fake_model_training_session
        tol = {"float32": 1e-5, "float64": 1e-8}[model_dtype]
        orig_max_epochs = config.trainer.max_epochs
        new_max_epochs = orig_max_epochs + 5

        # == continue training for a few more epochs ==
        with tempfile.TemporaryDirectory() as new_tmpdir_1:
            new_config = config.copy()
            new_config.trainer.max_epochs = new_max_epochs
            with open_dict(new_config):
                new_config["hydra"] = {"run": {"dir": new_tmpdir_1}}
            new_config = OmegaConf.create(new_config)
            config_path = new_tmpdir_1 + "/newconf.yaml"
            OmegaConf.save(config=new_config, f=config_path)
            retcode = subprocess.run(
                [
                    "nequip-train",
                    "-cn",
                    "newconf",
                    f"++ckpt_path='{tmpdir}/last.ckpt'",
                ],
                cwd=new_tmpdir_1,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _check_and_print(retcode)

            nequip_module = self.load_nequip_module_from_checkpoint(f"{new_tmpdir_1}/last.ckpt")
            restart_val_metrics = nequip_module.val_metrics[0].metrics_values_epoch

            # == retrain from scratch up to new_max_epochs ==
            with tempfile.TemporaryDirectory() as new_tmpdir_2:
                new_config = config.copy()
                new_config.trainer.max_epochs = new_max_epochs
                with open_dict(new_config):
                    new_config["hydra"] = {"run": {"dir": new_tmpdir_2}}
                new_config = OmegaConf.create(new_config)
                config_path = new_tmpdir_2 + "/newconf.yaml"
                OmegaConf.save(config=new_config, f=config_path)

                # == Train model ==
                retcode = subprocess.run(
                    ["nequip-train", "-cn", "newconf"],
                    cwd=new_tmpdir_2,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                _check_and_print(retcode)

                nequip_module = self.load_nequip_module_from_checkpoint(
                    f"{new_tmpdir_2}/last.ckpt"
                )

                # we don't test `loss` because they are numerically annoying due to the MSE squaring

                # == test val metrics reproduced ==
                oneshot_val_metrics = nequip_module.val_metrics[0].metrics_values_epoch

                print(restart_val_metrics)
                print(oneshot_val_metrics)
                assert len(restart_val_metrics) == len(oneshot_val_metrics)
                assert all(
                    [
                        math.isclose(a, b, rel_tol=tol)
                        for a, b in zip(
                            restart_val_metrics.values(), oneshot_val_metrics.values()
                        )
                    ]
                ), [restart_val_metrics, oneshot_val_metrics]
