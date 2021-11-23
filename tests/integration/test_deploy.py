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
from nequip.train import Trainer
from nequip.ase import NequIPCalculator


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_deploy(nequip_dataset, BENCHMARK_ROOT, device):
    dtype = str(torch.get_default_dtype())[len("torch.") :]

    # if torch.cuda.is_available():
    #     # TODO: is this true?
    #     pytest.skip("CUDA and subprocesses have issues")

    config_path = pathlib.Path(__file__).parents[2] / "configs/minimal.yaml"
    true_config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save time
        run_name = "test_deploy" + dtype
        root = "./"
        true_config["run_name"] = run_name
        true_config["root"] = root
        true_config["dataset_file_name"] = str(
            BENCHMARK_ROOT / "aspirin_ccsd-train.npz"
        )
        true_config["default_dtype"] = dtype
        true_config["max_epochs"] = 1
        true_config["n_train"] = 1
        true_config["n_val"] = 1
        config_path = "conf.yaml"
        with open(f"{tmpdir}/{config_path}", "w+") as fp:
            yaml.dump(true_config, fp)
        # Train model
        retcode = subprocess.run(["nequip-train", str(config_path)], cwd=tmpdir)
        retcode.check_returncode()
        # Deploy
        deployed_path = pathlib.Path(f"deployed_{dtype}.pth")
        retcode = subprocess.run(
            ["nequip-deploy", "build", f"{root}/{run_name}/", str(deployed_path)],
            cwd=tmpdir,
        )
        retcode.check_returncode()
        deployed_path = tmpdir / deployed_path
        assert deployed_path.is_file(), "Deploy didn't create file"

        # now test predictions the same
        best_mod, _ = Trainer.load_model_from_training_session(
            traindir=f"{tmpdir}/{root}/{run_name}/",
            model_name="best_model.pth",
            device=device,
        )
        best_mod.eval()

        data = AtomicData.to_AtomicDataDict(nequip_dataset[0].to(device))
        # Needed because of debug mode:
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = data[
            AtomicDataDict.TOTAL_ENERGY_KEY
        ].unsqueeze(0)
        train_pred = best_mod(data)[AtomicDataDict.TOTAL_ENERGY_KEY].to("cpu")

        # load model and check that metadata saved
        # TODO: use both CPU and CUDA to load?
        deploy_mod, metadata = deploy.load_deployed_model(
            deployed_path,
            device="cpu",
            set_global_options=False,  # don't need this corrupting test environment
        )
        # Everything we store right now is ASCII, so decode for printing
        assert metadata[deploy.NEQUIP_VERSION_KEY] == nequip.__version__
        assert np.allclose(float(metadata[deploy.R_MAX_KEY]), true_config["r_max"])
        assert len(metadata[deploy.TYPE_NAMES_KEY].split(" ")) == 3  # C, H, O

        data_idx = 0
        data = AtomicData.to_AtomicDataDict(nequip_dataset[data_idx].to("cpu"))
        deploy_pred = deploy_mod(data)[AtomicDataDict.TOTAL_ENERGY_KEY]
        assert torch.allclose(train_pred, deploy_pred, atol=1e-7)

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

        # Finally, try to load in ASE
        calc = NequIPCalculator.from_deployed_model(
            deployed_path,
            device="cpu",
            species_to_type_name={s: s for s in ("C", "H", "O")},
        )
        # use .get() so it's not transformed
        atoms = nequip_dataset.get(data_idx).to_ase()
        atoms.calc = calc
        ase_forces = atoms.get_potential_energy()
        assert torch.allclose(train_pred, torch.as_tensor(ase_forces), atol=1e-7)
