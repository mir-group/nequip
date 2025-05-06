import math
import torch

import tempfile
import subprocess
from omegaconf import OmegaConf, open_dict
import hydra

from conftest import _check_and_print


def load_nequip_module_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    training_module = hydra.utils.get_class(
        checkpoint["hyper_parameters"]["info_dict"]["training_module"]["_target_"]
    )
    nequip_module = training_module.load_from_checkpoint(checkpoint_path)
    return nequip_module


def test_batch_invariance(fake_model_training_session):
    # This actually tests two features:
    # 1. reproducibility of training with the same data, model and global seeds (we train in exactly the same way as in fake_model_training_session)
    # 2. invariance of validation metrics with the validation batch sizes

    # NOTE: at the time the tests were written, both minimal configs have val dataloader batch sizes of 5, so we only train again with batch_size=1 here
    config, tmpdir, env, model_dtype = fake_model_training_session
    tol = {"float32": 1e-5, "float64": 1e-8}[model_dtype]

    # == get necessary info from checkpoint ==
    nequip_module = load_nequip_module_from_checkpoint(f"{tmpdir}/last.ckpt")
    orig_train_loss = nequip_module.loss.metrics_values_epoch
    batchsize5_val_metrics = nequip_module.val_metrics[0].metrics_values_epoch

    # == train again with validation batch size 1 ==
    with tempfile.TemporaryDirectory() as new_tmpdir:
        new_config = config.copy()
        new_config.data.val_dataloader.batch_size = 1
        with open_dict(new_config):
            new_config["hydra"] = {"run": {"dir": new_tmpdir}}
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
        nequip_module = load_nequip_module_from_checkpoint(f"{new_tmpdir}/last.ckpt")

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
def test_restarts(fake_model_training_session):
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

        nequip_module = load_nequip_module_from_checkpoint(f"{new_tmpdir_1}/last.ckpt")
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

            nequip_module = load_nequip_module_from_checkpoint(
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
