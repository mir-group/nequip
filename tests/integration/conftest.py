import pytest
import tempfile
import subprocess
import hydra
import math
import torch
from omegaconf import OmegaConf, open_dict

from nequip.utils.unittests.utils import _training_session, _check_and_print


try:
    import schedulefree  # noqa: F401

    _SCHEDULEFREE_INSTALLED = True
except ImportError:
    _SCHEDULEFREE_INSTALLED = False


# available training modules for integration tests
# to save time, we only check the EMALightningModule
_ALL_TRAINING_MODULES = [
    # "nequip.train.NequIPLightningModule",
    "nequip.train.EMALightningModule",
    # "nequip.train.ConFIGLightningModule",
    # "nequip.train.EMAConFIGLightningModule",
]
if _SCHEDULEFREE_INSTALLED:
    _ALL_TRAINING_MODULES.append("nequip.train.ScheduleFreeLightningModule")


@pytest.fixture(
    scope="session",
    params=["minimal_aspirin.yaml", "minimal_toy_emt.yaml"],
)
def conffile(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        *(
            [
                {
                    "_target_": "nequip.train.ScheduleFreeLightningModule",
                    "optimizer": {
                        "_target_": "schedulefree.AdamWScheduleFree",
                        "lr": 0.01,
                        "weight_decay": 0.0,
                        "warmup_steps": 10,
                    },
                }
            ]
            if _SCHEDULEFREE_INSTALLED
            else []
        ),
        {"_target_": "nequip.train.EMALightningModule"},
        # {"_target_": "nequip.train.EMAConFIGLightningModule"},
    ],
)
def training_module_override_dict(request):
    return request.param


@pytest.fixture(scope="session", params=["fresh", "checkpoint", "package"])
def extra_train_from_save(request):
    """
    Whether the checkpoints for the tests come from a fresh model, `ModelFromCheckpoint`, or `ModelFromPackage`.
    """
    return request.param


@pytest.fixture(scope="session")
def fake_model_training_session(
    conffile, training_module_override_dict, model_dtype, extra_train_from_save
):
    session = _training_session(
        conffile,
        model_dtype,
        extra_train_from_save=extra_train_from_save,
        training_module_override_dict=training_module_override_dict,
    )
    config, tmpdir, env = next(session)
    yield config, tmpdir, env, model_dtype
    del session


class TrainingInvarianceBaseTest:
    # Override this in subclasses to specify which training modules to test
    _TRAINING_MODULES_TO_TEST = _ALL_TRAINING_MODULES

    def modify_model_config(self, original_config):
        raise NotImplementedError

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

        # Only test with allowed training modules
        current_training_module = config.training_module._target_
        if current_training_module not in self._TRAINING_MODULES_TO_TEST:
            return
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
            nequip_module = self.load_nequip_module_from_checkpoint(
                f"{new_tmpdir}/last.ckpt"
            )

            # == test training loss reproduced ==
            new_train_loss = nequip_module.loss.metrics_values_epoch
            print(orig_train_loss)
            print(new_train_loss)
            assert len(orig_train_loss) == len(new_train_loss)
            for name, orig_val, new_val in zip(
                orig_train_loss.keys(),
                orig_train_loss.values(),
                new_train_loss.values(),
            ):
                if not math.isclose(orig_val, new_val, rel_tol=tol):
                    raise AssertionError(
                        f"Training loss mismatch for '{name}': "
                        f"original={orig_val}, new={new_val}, "
                        f"diff={abs(orig_val - new_val)}, rel_tol={tol}"
                    )

            # == test val metrics invariance to batch size ==
            batchsize1_val_metrics = nequip_module.val_metrics[0].metrics_values_epoch
            print(batchsize5_val_metrics)
            print(batchsize1_val_metrics)
            assert len(batchsize5_val_metrics) == len(batchsize1_val_metrics)
            for name, batch5_val, batch1_val in zip(
                batchsize5_val_metrics.keys(),
                batchsize5_val_metrics.values(),
                batchsize1_val_metrics.values(),
            ):
                # do not include maxabserr or total energy in testing (per atom energy tested)
                if ("maxabserr" in name) or ("total_energy" in name):
                    continue
                if not math.isclose(batch5_val, batch1_val, rel_tol=tol, abs_tol=tol):
                    raise AssertionError(
                        f"Validation metric mismatch for '{name}': "
                        f"batch_size=5 value={batch5_val}, batch_size=1 value={batch1_val}, "
                        f"diff={abs(batch5_val - batch1_val)}, rel_tol={tol}"
                    )

    # TODO: will fail if train dataloader has shuffle=True
    # NOTE: test_restarts doesn't pass with PyTorch Lightning >= 2.6.0
    # may be related: https://github.com/Lightning-AI/pytorch-lightning/issues/20204
    # CI enforces lightning < 2.6.0 until this is resolved.
    def test_restarts(self, fake_model_training_session):
        config, tmpdir, env, model_dtype = fake_model_training_session

        # Only test with allowed training modules
        current_training_module = config.training_module._target_
        if current_training_module not in self._TRAINING_MODULES_TO_TEST:
            return
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

            nequip_module = self.load_nequip_module_from_checkpoint(
                f"{new_tmpdir_1}/last.ckpt"
            )
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
                for name, restart_val, oneshot_val in zip(
                    restart_val_metrics.keys(),
                    restart_val_metrics.values(),
                    oneshot_val_metrics.values(),
                ):
                    # do not include maxabserr or total energy in testing (per atom energy tested)
                    if ("maxabserr" in name) or ("total_energy" in name):
                        continue
                    if not math.isclose(
                        restart_val, oneshot_val, rel_tol=tol, abs_tol=tol
                    ):
                        raise AssertionError(
                            f"Validation metric mismatch for '{name}': "
                            f"restart value={restart_val}, oneshot value={oneshot_val}, "
                            f"diff={abs(restart_val - oneshot_val)}, rel_tol={tol}"
                        )
