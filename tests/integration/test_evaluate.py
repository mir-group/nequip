import pytest
import tempfile
import pathlib
import yaml
import subprocess
import os
import textwrap
import shutil

import numpy as np
import ase.io

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
        true_config["root"] = "./"
        true_config["dataset_file_name"] = str(
            BENCHMARK_ROOT / "aspirin_ccsd-train.npz"
        )
        true_config["default_dtype"] = dtype
        true_config["max_epochs"] = 2
        true_config["model_builders"] = [builder]
        # We need truth labels as inputs for these fake testing models
        true_config["_override_allow_truth_label_inputs"] = True

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
        retcode = subprocess.run(["nequip-train", "conf.yaml"], cwd=tmpdir, env=env)
        retcode.check_returncode()

        yield builder, true_config, tmpdir, env


@pytest.mark.parametrize("do_test_idcs", [True, False])
@pytest.mark.parametrize("do_metrics", [True, False])
@pytest.mark.parametrize("do_output_fields", [True, False])
def test_metrics(training_session, do_test_idcs, do_metrics, do_output_fields):

    builder, true_config, tmpdir, env = training_session
    # == Run test error ==
    outdir = f"{true_config['root']}/{true_config['run_name']}/"

    default_params = {
        "train-dir": outdir,
        "output": "out.xyz",
        "log": "out.log",
    }

    def runit(params: dict):
        tmp = default_params.copy()
        tmp.update(params)
        params = tmp
        del tmp
        retcode = subprocess.run(
            ["nequip-evaluate"]
            + sum(
                (["--" + k, str(v)] for k, v in params.items() if v is not None),
                [],
            ),
            cwd=tmpdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        retcode.check_returncode()

        # Check the output
        metrics = dict(
            [
                tuple(e.strip() for e in line.split("=", 1))
                for line in retcode.stdout.decode().splitlines()
            ]
        )
        metrics = {k: float(v) for k, v in metrics.items()}
        return metrics

    # Test idcs
    if do_test_idcs:
        # The Aspirin dataset is 1000 frames long
        # Pick some arbitrary number of frames
        test_idcs_arr = torch.randperm(1000)[:257]
        test_idcs = "some-test-idcs.pth"
        torch.save(test_idcs_arr, f"{tmpdir}/{test_idcs}")
    else:
        test_idcs = None  # ignore and use default
    default_params["test-indexes"] = test_idcs

    # Metrics
    if do_metrics:
        # Write an explicit metrics file
        metrics_yaml = "my-metrics.yaml"
        with open(f"{tmpdir}/{metrics_yaml}", "w") as f:
            # Write out a fancier metrics file
            # We don't use PerSpecies here since the simple models don't fill ATOM_TYPE_KEY right now
            # ^ TODO!
            f.write(
                textwrap.dedent(
                    """
                    metrics_components:
                      - - forces
                        - rmse
                        - report_per_component: True
                    """
                )
            )
        expect_metrics = {"f_rmse_0", "f_rmse_1", "f_rmse_2"}
    else:
        metrics_yaml = None
        # Regardless of builder, with minimal.yaml, we should have RMSE and MAE
        expect_metrics = {"f_mae", "f_rmse"}
    default_params["metrics-config"] = metrics_yaml

    if do_output_fields:
        output_fields = [AtomicDataDict.NODE_FEATURES_KEY]
        default_params["output-fields"] = ",".join(output_fields)
    else:
        output_fields = None

    # -- First run --
    metrics = runit({"train-dir": outdir, "batch-size": 200, "device": "cpu"})
    # move out.xyz to out-orig.xyz
    shutil.move(tmpdir + "/out.xyz", tmpdir + "/out-orig.xyz")
    # Load it
    orig_atoms = ase.io.read(tmpdir + "/out-orig.xyz", index=":", format="extxyz")

    # check that we have the metrics
    assert set(metrics.keys()) == expect_metrics

    # check metrics
    if builder == IdentityModel:
        for metric, err in metrics.items():
            assert np.allclose(err, 0.0), f"Metric `{metric}` wasn't zero!"
    elif builder == ConstFactorModel:
        # TODO: check comperable to naive numpy compute
        pass

    # check we got output fields
    if output_fields is not None:
        for a in orig_atoms:
            for key in output_fields:
                if key == AtomicDataDict.NODE_FEATURES_KEY:
                    assert a.arrays[AtomicDataDict.NODE_FEATURES_KEY].shape == (
                        len(a),
                        3,  # THIS IS SPECIFIC TO THE HACK IN ConstFactorModel and friends
                    )
                else:
                    raise RuntimeError

    # -- Check insensitive to batch size --
    for batch_size in (13, 1000):
        metrics2 = runit(
            {
                "train-dir": outdir,
                "batch-size": batch_size,
                "device": "cpu",
                "output": f"{batch_size}.xyz",
                "log": f"{batch_size}.log",
            }
        )
        for k, v in metrics.items():
            assert np.all(np.abs(v - metrics2[k]) < 1e-5)

        # Check the output XYZ
        batch_atoms = ase.io.read(tmpdir + "/out-orig.xyz", index=":", format="extxyz")
        for origframe, newframe in zip(orig_atoms, batch_atoms):
            assert np.allclose(origframe.get_positions(), newframe.get_positions())
            assert np.array_equal(
                origframe.get_atomic_numbers(), newframe.get_atomic_numbers()
            )
            assert np.array_equal(origframe.get_pbc(), newframe.get_pbc())
            assert np.array_equal(origframe.get_cell(), newframe.get_cell())
            if output_fields is not None:
                for key in output_fields:
                    # TODO handle info fields too
                    assert np.allclose(origframe.arrays[key], newframe.arrays[key])

    # Check GPU
    if torch.cuda.is_available():
        metrics_gpu = runit({"train-dir": outdir, "batch-size": 17, "device": "cuda"})
        for k, v in metrics.items():
            assert np.all(np.abs(v - metrics_gpu[k]) < 1e-3)  # GPU nondeterminism
