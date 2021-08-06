import pytest
import tempfile
import pathlib
import yaml
import subprocess
import os

import numpy as np
import torch

from nequip.data import AtomicDataDict

from test_train import ConstFactorModel, IdentityModel  # noqa


@pytest.fixture(
    scope="module",
    params=[
        ("minimal.yaml", AtomicDataDict.FORCE_KEY),
    ],
)
def conffile(request):
    return request.param


@pytest.fixture(scope="module", params=[ConstFactorModel, IdentityModel])
def training_session(request, BENCHMARK_ROOT, conffile):
    conffile, _ = conffile
    builder = request.param
    dtype = str(torch.get_default_dtype())[len("torch.") :]

    # if torch.cuda.is_available():
    #     # TODO: is this true?
    #     pytest.skip("CUDA and subprocesses have issues")

    path_to_this_file = pathlib.Path(__file__)
    config_path = path_to_this_file.parents[2] / f"configs/{conffile}"
    true_config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    with tempfile.TemporaryDirectory() as tmpdir:
        # == Run training ==
        # Save time
        run_name = "test_train_" + dtype
        true_config["run_name"] = run_name
        true_config["root"] = tmpdir
        true_config["dataset_file_name"] = str(
            BENCHMARK_ROOT / "aspirin_ccsd-train.npz"
        )
        true_config["default_dtype"] = dtype
        true_config["max_epochs"] = 2
        true_config["model_builder"] = builder

        # to be a true identity, we can't have rescaling
        true_config["global_rescale_shift"] = None
        true_config["global_rescale_scale"] = None

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
            ["nequip-train", str(config_path)], cwd=tmpdir, env=env
        )
        retcode.check_returncode()

        yield builder, true_config, tmpdir, env


def test_metrics(training_session):
    builder, true_config, tmpdir, env = training_session
    # == Run test error ==
    outdir = f"{true_config['root']}/{true_config['run_name']}/"

    retcode = subprocess.run(
        ["nequip-test-error", "--train-dir", outdir],
        cwd=tmpdir,
        env=env,
        stdout=subprocess.PIPE,
    )
    retcode.check_returncode()

    # Check the output
    metrics = dict(
        [
            tuple(e.strip() for e in line.split("="))
            for line in retcode.stdout.decode().splitlines()
        ]
    )
    metrics = {tuple(k.split("_")): float(v) for k, v in metrics.items()}

    # Regardless of builder, with minimal.yaml, we should have RMSE and MAE
    assert set(metrics.keys()) == {("forces", "mae"), ("forces", "rmse")}

    if builder == IdentityModel:
        for metric, err in metrics.items():
            assert np.allclose(err, 0.0), f"Metric `{metric}` wasn't zero!"
    elif builder == ConstFactorModel:
        pass
        # TODO: check against naive numpy metrics
