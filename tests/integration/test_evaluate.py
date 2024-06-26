import pytest
import subprocess
import textwrap
import shutil

import numpy as np
import ase.io

import torch

from nequip.data import AtomicDataDict

from conftest import IdentityModel, ConstFactorModel, _check_and_print


@pytest.mark.parametrize("do_test_idcs", [True, False])
@pytest.mark.parametrize("do_metrics", [True, False])
@pytest.mark.parametrize("do_output_fields", [True, False])
def test_metrics(
    fake_model_training_session, conffile, do_test_idcs, do_metrics, do_output_fields
):
    energy_only: bool = conffile[0] == "minimal_eng.yaml"
    if energy_only:
        # By default, don't run the energy only tests... they are redundant and add a _lot_ of expense
        pytest.skip()
    builder, true_config, tmpdir, env = fake_model_training_session
    if builder not in (IdentityModel, ConstFactorModel):
        pytest.skip()
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
        _check_and_print(retcode)

        # Check the output
        metrics = dict(
            [
                tuple(e.strip() for e in line.split("=", 1))
                for line in retcode.stderr.decode().splitlines()
                if " = " in line
            ]
        )
        metrics = {k: float(v) for k, v in metrics.items()}
        return metrics

    # Test idcs
    if do_test_idcs:
        if conffile[0] == "minimal.yaml":
            # The Aspirin dataset is 1000 frames long
            # Pick some arbitrary number of frames
            test_idcs_arr = torch.randperm(1000)[:257]
        elif conffile[0] == "minimal_toy_emt.yaml":
            # The toy EMT dataset is 50 frames long
            # Pick some arbitrary number of frames
            test_idcs_arr = torch.randperm(50)[:7]
        else:
            raise KeyError
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
            if energy_only:
                f.write(
                    textwrap.dedent(
                        """
                        metrics_components:
                          - - total_energy
                            - mae
                          - - total_energy
                            - mae
                            - PerAtom: True
                          - - total_energy
                            - mae
                            - stratify: 10%_range
                              PerAtom: True
                        """
                    )
                )
                expect_metrics = {
                    "e_mae",
                    "e/N_mae",
                    "10%-20%_range_e/N_mae",
                }
            else:
                # Write out a fancier metrics file
                f.write(
                    textwrap.dedent(
                        """
                        metrics_components:
                          - - forces
                            - rmse
                            - report_per_component: True
                          - - forces
                            - mae
                            - PerSpecies: True
                          - - total_energy
                            - mae
                          - - total_energy
                            - mae
                            - PerAtom: True
                          - - total_energy
                            - mae
                            - stratify: 10%_range
                              PerAtom: True
                          - - forces
                            - rmse
                            - stratify: 10%_population
                          - - total_energy
                            - mae
                            - stratify: 0.5
                        """
                    )
                )
                expect_metrics = {
                    "f_rmse_0",
                    "f_rmse_1",
                    "f_rmse_2",
                    "psavg_f_mae",
                    "e_mae",
                    "e/N_mae",
                    "10%-20%_range_e/N_mae",
                    "30%-40%_population_f_rmse",
                    "0.5-1.0_range_e_mae",
                }.union(
                    {
                        # For the PerSpecies
                        sym + "_f_mae"
                        for sym in true_config["chemical_symbols"]
                    }
                )
    else:
        metrics_yaml = None
        # Regardless of builder, with minimal.yaml, we should have RMSE and MAE
        if energy_only:
            expect_metrics = {"e_mae", "e_rmse"}
        else:
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
    assert expect_metrics.issubset(set(metrics.keys()))

    # check metrics
    if builder == IdentityModel:
        true_identity: bool = true_config["default_dtype"] == true_config["model_dtype"]
        for metric, err in metrics.items():
            if not np.isnan(err):
                # see test_train.py for discussion
                assert np.allclose(
                    err,
                    0.0,
                    atol=(
                        1e-8 if true_identity else (1e-2 if "_e" in metric else 1e-4)
                    ),
                ), f"Metric `{metric}` wasn't zero!"
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
            if not np.isnan(v):
                assert np.allclose(
                    v,
                    metrics2[k],
                    atol={
                        torch.float32: 1e-6
                        + (1e-1 if "population_f_rmse" in k else 0.0),
                        torch.float64: 1e-8,
                    }[torch.get_default_dtype()],
                )
            else:
                assert np.isnan(metrics2[k])  # assert both are nans

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
            assert np.allclose(
                v,
                metrics_gpu[k],
                atol={torch.float32: 1e-4, torch.float64: 1e-6}[
                    torch.get_default_dtype()
                ],
            )
