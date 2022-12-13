import pytest
import tempfile
import pathlib
import yaml
import subprocess
import os

import numpy as np
import torch

from test_train import LearningFactorModel, _check_and_print


@pytest.mark.parametrize(
    "conffile",
    [
        "minimal.yaml",
    ],
)
@pytest.mark.parametrize("builder", [LearningFactorModel, None])
def test_metrics(nequip_dataset, BENCHMARK_ROOT, conffile, builder):

    dtype = str(torch.get_default_dtype())[len("torch.") :]

    device = "cpu"
    num_worker = 4
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = "cuda"
        num_worker = torch.cuda.device_count()

    path_to_this_file = pathlib.Path(__file__)
    config_path = path_to_this_file.parents[2] / f"configs/{conffile}"
    true_config = yaml.load(config_path.read_text(), Loader=yaml.Loader)

    with tempfile.TemporaryDirectory() as tmpdir:
        # setup config
        run_name_true = "test_train_" + dtype
        true_config["run_name"] = run_name_true
        true_config["root"] = "./"
        true_config["dataset_file_name"] = str(
            BENCHMARK_ROOT / "aspirin_ccsd-train.npz"
        )
        true_config["default_dtype"] = dtype
        true_config["device"] = device
        true_config["batch_size"] = num_worker * 2
        true_config["validation_batch_size"] = num_worker * 3
        # for training to be the same, it must be a multiple
        # otherwise the distributed sampler has to do repeats / drops
        # TODO: test this with and without a +1 and just ignore training metrics with the +1
        true_config["n_train"] = num_worker * 6
        # for validation, it should correctly handle any arbitrary number
        # TODO: make that so
        true_config["n_val"] = num_worker * 6
        true_config["max_epochs"] = 3
        true_config["seed"] = 950590
        true_config["dataset_seed"] = 34345
        # important so that both runs have the same presentation order
        # in theory just the seeds should handle this, but who knows...
        true_config["shuffle"] = False
        if builder is not None:
            # We just don't add rescaling:
            true_config["model_builders"] = [builder]
        # We need truth labels as inputs for these fake testing models
        true_config["_override_allow_truth_label_inputs"] = True

        distributed_config = true_config.copy()
        run_name_distributed = "test_train_distributed_" + dtype
        distributed_config["run_name"] = run_name_distributed

        config_path_true = tmpdir + "/conf_true.yaml"
        config_path_distributed = tmpdir + "/conf_distributed.yaml"
        with open(config_path_true, "w+") as fp:
            yaml.dump(true_config, fp)
        with open(config_path_distributed, "w+") as fp:
            yaml.dump(distributed_config, fp)

        env = dict(os.environ)
        # make this script available so model builders can be loaded
        env["PYTHONPATH"] = ":".join(
            [str(path_to_this_file.parent)] + env.get("PYTHONPATH", "").split(":")
        )

        # == run distributed FIRST to make it have to process dataset ==
        nequip_train_script_path = (
            subprocess.check_output("which nequip-train", shell=True)
            .decode("ascii")
            .strip()
        )
        retcode = subprocess.run(
            [
                "torchrun",  # TODO
                "--nnodes",
                "1",
                "--nproc_per_node",
                str(num_worker),
                nequip_train_script_path,
                "conf_distributed.yaml",
                "--distributed",
            ],
            cwd=tmpdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _check_and_print(retcode)

        # == Train truth model ==
        retcode = subprocess.run(
            ["nequip-train", "conf_true.yaml"],
            cwd=tmpdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _check_and_print(retcode)

        # == Load metrics ==
        outdir_true = f"{tmpdir}/{true_config['root']}/{run_name_true}/"
        outdir_distributed = f"{tmpdir}/{true_config['root']}/{run_name_distributed}/"

        # epoch metrics
        # only epoch metrics are synced
        dat_true, dat_distributed = [
            np.genfromtxt(
                f"{outdir}/metrics_epoch.csv",
                delimiter=",",
                names=True,
                dtype=None,
            )
            for outdir in (outdir_true, outdir_distributed)
        ]

        for key in dat_true.dtype.names:
            if key in {"wall"}:
                continue
            assert np.allclose(dat_true[key], dat_distributed[key])
