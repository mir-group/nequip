import math
import torch
import pytest

import tempfile
import subprocess
from omegaconf import OmegaConf, open_dict
import hydra

from conftest import _check_and_print

def load_nequip_module_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cuda",
        weights_only=False,
    )
    training_module = hydra.utils.get_class(
        checkpoint["hyper_parameters"]["info_dict"]["training_module"]["_target_"]
    )
    nequip_module = training_module.load_from_checkpoint(checkpoint_path)
    return nequip_module

def create_oeq_config(config, new_tmpdir):
    new_config = config.copy()
    with open_dict(new_config):
        new_config["hydra"] = {"run": {"dir": new_tmpdir}}

        training_module = new_config["training_module"]
        original_model = training_module["model"]
        training_module["model"] = {
            "_target_": "nequip.model.modify",
            "modifiers": [{"modifier": "enable_OpenEquivariance"}],
            "model": original_model,
        }
    new_config = OmegaConf.create(new_config)
    return new_config

def test_oeq_training(fake_model_training_session):
    # Test that an OEQ model reproduces the training loss 

    # NOTE: at the time the tests were written, both minimal configs have val dataloader batch sizes of 5, so we only train again with batch_size=1 here
    config, tmpdir, env, model_dtype = fake_model_training_session
    tol = {"float32": 1e-5, "float64": 1e-8}[model_dtype]

    # == get necessary info from checkpoint ==
    nequip_module = load_nequip_module_from_checkpoint(f"{tmpdir}/last.ckpt")
    orig_train_loss = nequip_module.loss.metrics_values_epoch 

    # == train again with validation batch size 1 ==
    with tempfile.TemporaryDirectory() as new_tmpdir:
        new_config = create_oeq_config(config, new_tmpdir)
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