import pytest
import tempfile
import pathlib
import yaml
import subprocess
import os

import numpy as np
import torch

from nequip.data import AtomicDataDict

from conftest import (
    IdentityModel,
    ConstFactorModel,
    LearningFactorModel,
    _check_and_print,
)


def test_metrics(fake_model_training_session, model_dtype):
    default_dtype = str(torch.get_default_dtype()).lstrip("torch.")
    builder, true_config, tmpdir, env = fake_model_training_session

    # == Load metrics ==
    outdir = f"{tmpdir}/{true_config['root']}/{true_config['run_name']}/"

    if builder == IdentityModel or builder == LearningFactorModel:
        for which in ("train", "val"):

            dat = np.genfromtxt(
                f"{outdir}/metrics_batch_{which}.csv",
                delimiter=",",
                names=True,
                dtype=None,
            )
            for field in dat.dtype.names:
                if field == "epoch" or field == "batch":
                    continue
                # Everything else should be a loss or a metric
                if builder == IdentityModel:
                    if model_dtype == default_dtype:
                        # We have a true identity model
                        assert np.allclose(
                            dat[field],
                            0.0,
                            atol=1e-6 if default_dtype == "float32" else 1e-9,
                        ), f"Loss/metric `{field}` wasn't all zeros for {which}"
                    else:
                        # we have an approximate identity model that applies a floating point truncation
                        # in the actual aspirin test data used here, the truncation error is maximally 0.0155
                        # there is also no rescaling so everything is in real units here
                        assert np.all(
                            dat[field] < 0.02
                        ), f"Loss/metric `{field}` wasn't approximately zeros for {which}"
                elif builder == LearningFactorModel:
                    assert (
                        dat[field][-1] < dat[field][0]
                    ), f"Loss/metric `{field}` didn't go down for {which}"

    # epoch metrics
    dat = np.genfromtxt(
        f"{outdir}/metrics_epoch.csv",
        delimiter=",",
        names=True,
        dtype=None,
    )
    for field in dat.dtype.names:
        if field == "epoch" or field == "wall" or field == "LR":
            continue

        # Everything else should be a loss or a metric
        if builder == IdentityModel:
            if model_dtype == default_dtype:
                # we have a true identity model
                assert np.allclose(
                    dat[field][1:],
                    0.0,
                    atol=1e-6 if default_dtype == "float32" else 1e-9,
                ), f"Loss/metric `{field}` wasn't all equal to zero for epoch"
            else:
                # we have an approximate identity model that applies a floating point truncation
                # see above
                assert np.all(
                    dat[field][1:] < 0.02
                ), f"Loss/metric `{field}` wasn't approximately zeros for {which}"
        elif builder == ConstFactorModel:
            # otherwise just check its constant.
            # epoch-wise numbers should be the same, since there's no randomness at this level
            assert np.allclose(
                dat[field], dat[field][0]
            ), f"Loss/metric `{field}` wasn't all equal to {dat[field][0]} for epoch"
        elif builder == LearningFactorModel:
            assert (
                dat[field][-1] < dat[field][0]
            ), f"Loss/metric `{field}` didn't go down across epochs"

    # == Check model ==
    model = torch.load(outdir + "/last_model.pth")

    if builder == IdentityModel:
        # GraphModel.IdentityModel
        zero = model["model.zero"]
        # Since the loss is always zero, even though the constant
        # 1 was trainable, it shouldn't have changed
        # the tolerances when loss is nonzero are large-ish because the default learning rate 0.01 is high
        # these tolerances are _also_ in real units
        assert torch.allclose(
            zero,
            torch.zeros(1, device=zero.device, dtype=zero.dtype),
            atol=1e-7 if model_dtype == default_dtype else 1e-2,
        )


@pytest.mark.parametrize(
    "conffile",
    [
        "minimal.yaml",
        "minimal_eng.yaml",
    ],
)
def test_requeue(nequip_dataset, BENCHMARK_ROOT, conffile):
    # TODO test metrics against one that goes all the way through
    builder = IdentityModel  # TODO: train a real model?
    dtype = str(torch.get_default_dtype())[len("torch.") :]

    path_to_this_file = pathlib.Path(__file__)
    config_path = path_to_this_file.parents[2] / f"configs/{conffile}"
    true_config = yaml.load(config_path.read_text(), Loader=yaml.Loader)

    with tempfile.TemporaryDirectory() as tmpdir:

        run_name = "test_requeue_" + dtype
        true_config["run_name"] = run_name
        true_config["append"] = True
        true_config["root"] = "./"
        true_config["dataset_file_name"] = str(
            BENCHMARK_ROOT / "aspirin_ccsd-train.npz"
        )
        true_config["default_dtype"] = dtype
        # We just don't add rescaling:
        true_config["model_builders"] = [builder]
        # We need truth labels as inputs for these fake testing models
        true_config["model_input_fields"] = {
            AtomicDataDict.FORCE_KEY: "1o",
            AtomicDataDict.TOTAL_ENERGY_KEY: "0e",
        }

        for irun in range(3):

            true_config["max_epochs"] = 2 * (irun + 1)
            config_path = tmpdir + "/conf.yaml"
            with open(config_path, "w+") as fp:
                yaml.dump(true_config, fp)

            # == Train model ==
            env = dict(os.environ)
            # make this script available so model builders can be loaded
            env["PYTHONPATH"] = ":".join(
                [str(path_to_this_file.parent)] + env.get("PYTHONPATH", "").split(":")
            )

            retcode = subprocess.run(
                # Supress the warning cause we use general config for all the fake models
                ["nequip-train", "conf.yaml", "--warn-unused"],
                cwd=tmpdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _check_and_print(retcode)

            # == Load metrics ==
            dat = np.genfromtxt(
                f"{tmpdir}/{run_name}/metrics_epoch.csv",
                delimiter=",",
                names=True,
                dtype=None,
            )

            assert len(dat["epoch"]) == true_config["max_epochs"]
