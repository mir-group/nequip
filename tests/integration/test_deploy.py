import pytest
import tempfile
import pathlib
import yaml
import subprocess

import numpy as np
import torch

import nequip
from nequip.data import AtomicDataDict, AtomicData
from nequip.scripts import deploy


def test_deploy(nequip_dataset, BENCHMARK_ROOT):

    dtype = str(torch.get_default_dtype())[len("torch.") :]

    # if torch.cuda.is_available():
    #     # TODO: is this true?
    #     pytest.skip("CUDA and subprocesses have issues")

    config_path = pathlib.Path(__file__).parents[2] / "configs/minimal.yaml"
    true_config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save time
        run_name = "test_deploy" + dtype
        true_config["run_name"] = run_name
        true_config["root"] = tmpdir
        true_config["dataset_file_name"] = str(
            BENCHMARK_ROOT / "aspirin_ccsd-train.npz"
        )
        true_config["default_dtype"] = dtype
        true_config["max_epochs"] = 1
        true_config["n_train"] = 1
        true_config["n_val"] = 1
        config_path = tmpdir + "/conf.yaml"
        with open(config_path, "w+") as fp:
            yaml.dump(true_config, fp)
        # Train model
        retcode = subprocess.run(["nequip-train", str(config_path)], cwd=tmpdir)
        retcode.check_returncode()
        # Deploy
        deployed_path = tmpdir / pathlib.Path(f"deployed_{dtype}.pth")
        retcode = subprocess.run(
            ["nequip-deploy", "build", f"{tmpdir}/{run_name}/", str(deployed_path)],
            cwd=tmpdir,
        )
        retcode.check_returncode()
        assert deployed_path.is_file(), "Deploy didn't create file"

        # now test predictions the same
        best_mod = torch.load(f"{tmpdir}/{run_name}/best_model.pth")
        device = next(best_mod.parameters()).device
        data = AtomicData.to_AtomicDataDict(nequip_dataset[0].to(device))
        # Needed because of debug mode:
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = data[
            AtomicDataDict.TOTAL_ENERGY_KEY
        ].unsqueeze(0)
        train_pred = best_mod(data)[AtomicDataDict.TOTAL_ENERGY_KEY]

        # load model and check that metadata saved
        metadata = {
            deploy.NEQUIP_VERSION_KEY: "",
            deploy.R_MAX_KEY: "",
        }
        deploy_mod = torch.jit.load(
            deployed_path, _extra_files=metadata, map_location="cpu"
        )
        # Everything we store right now is ASCII, so decode for printing
        metadata = {k: v.decode("ascii") for k, v in metadata.items()}
        assert metadata[deploy.NEQUIP_VERSION_KEY] == nequip.__version__
        assert np.allclose(float(metadata[deploy.R_MAX_KEY]), true_config["r_max"])

        data = AtomicData.to_AtomicDataDict(nequip_dataset[0].to("cpu"))
        deploy_pred = deploy_mod(data)[AtomicDataDict.TOTAL_ENERGY_KEY]
        assert torch.allclose(train_pred.to("cpu"), deploy_pred, atol=1e-7)

        # now test info
        retcode = subprocess.run(
            ["nequip-deploy", "info", str(deployed_path)],
            text=True,
            stdout=subprocess.PIPE,
        )
        retcode.check_returncode()
        # Try to load extract config
        config = yaml.load(retcode.stdout, Loader=yaml.Loader)
        del config
