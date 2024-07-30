import tempfile

import subprocess
import os

import math

from nequip.train import NequIPLightningModule

from omegaconf import OmegaConf, open_dict

from conftest import _check_and_print


def test_batch_invariance(fake_model_training_session):
    # This actually tests two features:
    # 1. reproducibility of training with the same data, model and global seeds (we train in exactly the same way as in fake_model_training_session)
    # 2. invariance of validation metrics with the validation batch sizes

    # NOTE: at the time the tests were written, both minimal configs have val dataloader batch sizes of 5, so we only train again with batch_size=1 here
    config, tmpdir, env = fake_model_training_session
    tol = {"float32": 1e-5, "float64": 1e-8}[config.model.model_dtype]

    # == get necessary info from checkpoint ==
    nequip_module = NequIPLightningModule.load_from_checkpoint(f"{tmpdir}/last.ckpt")
    orig_train_loss = nequip_module.loss.metrics_values_epoch
    batchsize5_val_metrics = nequip_module.val_metrics[0].metrics_values_epoch

    # == train again with validation batch size 1 ==
    with tempfile.TemporaryDirectory() as new_tmpdir:
        new_config = config.copy()
        new_config.data.val_dataloader_kwargs.batch_size = 1
        with open_dict(new_config):
            new_config["hydra"] = {"run": {"dir": new_tmpdir}}
        new_config = OmegaConf.create(new_config)
        config_path = new_tmpdir + "/newconf.yaml"
        OmegaConf.save(config=new_config, f=config_path)

        # == Train model ==
        env = dict(os.environ)
        retcode = subprocess.run(
            ["nequip-train", "-cn", "newconf"],
            cwd=new_tmpdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _check_and_print(retcode)
        nequip_module = NequIPLightningModule.load_from_checkpoint(
            f"{new_tmpdir}/last.ckpt"
        )

        # == test training loss reproduced ==
        new_train_loss = nequip_module.loss.metrics_values_epoch
        assert len(orig_train_loss) == len(new_train_loss)
        assert all(
            [
                math.isclose(a, b, rel_tol=tol)
                for a, b in zip(orig_train_loss, new_train_loss)
            ]
        )

        # == test val metrics invariance to batch size ==
        batchsize1_val_metrics = nequip_module.val_metrics[0].metrics_values_epoch

        assert len(batchsize5_val_metrics) == len(batchsize1_val_metrics)
        assert all(
            [
                math.isclose(a, b, rel_tol=tol)
                for a, b in zip(batchsize5_val_metrics, batchsize1_val_metrics)
            ]
        )


# TODO: will fail if train dataloader has shuffle=True
def test_restarts(fake_model_training_session):
    config, tmpdir, env = fake_model_training_session
    tol = {"float32": 1e-5, "float64": 1e-8}[config.model.model_dtype]
    orig_max_epochs = config.train.trainer.max_epochs
    new_max_epochs = orig_max_epochs + 5

    # == continue training for a few more epochs ==
    with tempfile.TemporaryDirectory() as new_tmpdir_1:
        new_config = config.copy()
        new_config.train.trainer.max_epochs = new_max_epochs
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
            env=dict(os.environ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _check_and_print(retcode)

        nequip_module = NequIPLightningModule.load_from_checkpoint(
            f"{new_tmpdir_1}/last.ckpt"
        )
        restart_train_loss = nequip_module.loss.metrics_values_epoch
        restart_val_metrics = nequip_module.val_metrics[0].metrics_values_epoch

        # == retrain from scratch up to new_max_epochs ==
        with tempfile.TemporaryDirectory() as new_tmpdir_2:
            new_config = config.copy()
            new_config.train.trainer.max_epochs = new_max_epochs
            with open_dict(new_config):
                new_config["hydra"] = {"run": {"dir": new_tmpdir_2}}
            new_config = OmegaConf.create(new_config)
            config_path = new_tmpdir_2 + "/newconf.yaml"
            OmegaConf.save(config=new_config, f=config_path)

            # == Train model ==
            retcode = subprocess.run(
                ["nequip-train", "-cn", "newconf"],
                cwd=new_tmpdir_2,
                env=dict(os.environ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _check_and_print(retcode)
            nequip_module = NequIPLightningModule.load_from_checkpoint(
                f"{new_tmpdir_2}/last.ckpt"
            )

            # == test training loss reproduced ==
            oneshot_train_loss = nequip_module.loss.metrics_values_epoch

            print(restart_train_loss)
            print(oneshot_train_loss)
            assert len(restart_train_loss) == len(oneshot_train_loss)
            assert all(
                [
                    math.isclose(a, b, rel_tol=tol)
                    for a, b in zip(restart_train_loss, oneshot_train_loss)
                ]
            )

            # == test val metrics reproduced ==
            oneshot_val_metrics = nequip_module.val_metrics[0].metrics_values_epoch

            print(restart_val_metrics)
            print(oneshot_val_metrics)
            assert len(restart_val_metrics) == len(oneshot_val_metrics)
            assert all(
                [
                    math.isclose(a, b, rel_tol=tol)
                    for a, b in zip(restart_val_metrics, oneshot_val_metrics)
                ]
            )
