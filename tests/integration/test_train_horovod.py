import pytest
import tempfile
import pathlib
import yaml
import subprocess
import os

import numpy as np
import torch

from test_train import LearningFactorModel

hvd = pytest.importorskip("horovod.torch")


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
        true_config["n_train"] = num_worker * 6
        true_config["n_val"] = num_worker * 4
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

        horovod_config = true_config.copy()
        run_name_horovod = "test_train_horovod_" + dtype
        horovod_config["run_name"] = run_name_horovod

        config_path_true = tmpdir + "/conf_true.yaml"
        config_path_horovod = tmpdir + "/conf_horovod.yaml"
        with open(config_path_true, "w+") as fp:
            yaml.dump(true_config, fp)
        with open(config_path_horovod, "w+") as fp:
            yaml.dump(horovod_config, fp)

        env = dict(os.environ)
        # make this script available so model builders can be loaded
        env["PYTHONPATH"] = ":".join(
            [str(path_to_this_file.parent)] + env.get("PYTHONPATH", "").split(":")
        )

        # == run horovod FIRST to make it have to process dataset ==
        retcode = subprocess.run(
            [
                "horovodrun",
                "-np",
                str(num_worker),
                "-H",
                f"localhost:{num_worker}",
                "nequip-train",
                "--horovod",
                "conf_horovod.yaml",
            ],
            cwd=tmpdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        retcode.check_returncode()

        # == Train truth model ==
        retcode = subprocess.run(
            ["nequip-train", "conf_true.yaml"],
            cwd=tmpdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        retcode.check_returncode()

        # == Load metrics ==
        outdir_true = f"{tmpdir}/{true_config['root']}/{run_name_true}/"
        outdir_horovod = f"{tmpdir}/{true_config['root']}/{run_name_horovod}/"

        # epoch metrics
        # only epoch metrics are synced
        dat_true, dat_horovod = [
            np.genfromtxt(
                f"{outdir}/metrics_epoch.csv",
                delimiter=",",
                names=True,
                dtype=None,
            )
            for outdir in (outdir_true, outdir_horovod)
        ]

        for key in dat_true.dtype.names:
            if key in {"wall"}:
                continue
            assert np.allclose(dat_true[key], dat_horovod[key])
