import pytest
import tempfile
import pathlib
import yaml
import subprocess
import sys

import numpy as np
import torch

import nequip
from nequip.data import AtomicDataDict, AtomicData, dataset_from_config
from nequip.utils import Config
from nequip.scripts import deploy
from nequip.train import Trainer
from nequip.ase import NequIPCalculator


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_deploy(BENCHMARK_ROOT, device):
    dtype = str(torch.get_default_dtype())[len("torch.") :]
    atol = {"float32": 1e-5, "float64": 1e-7}[dtype]

    # if torch.cuda.is_available():
    #     # TODO: is this true?
    #     pytest.skip("CUDA and subprocesses have issues")

    keys = [
        AtomicDataDict.TOTAL_ENERGY_KEY,
        AtomicDataDict.FORCE_KEY,
        AtomicDataDict.PER_ATOM_ENERGY_KEY,
    ]

    config_path = pathlib.Path(__file__).parents[2] / "configs/minimal.yaml"
    true_config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save time
        run_name = "test_deploy" + dtype
        root = tmpdir + "/nequip_rootdir/"
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
        full_config_path = f"{tmpdir}/{config_path}"
        with open(full_config_path, "w+") as fp:
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
            traindir=f"{root}/{run_name}/",
            model_name="best_model.pth",
            device=device,
        )
        best_mod.eval()

        # load train dataset, already cached
        dataset = dataset_from_config(Config.from_file(full_config_path))
        data = AtomicData.to_AtomicDataDict(dataset[0].to(device))
        for k in keys:
            data.pop(k, None)
        train_pred = best_mod(data)
        train_pred = {k: train_pred[k].to("cpu") for k in keys}

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
        data = AtomicData.to_AtomicDataDict(dataset[data_idx].to("cpu"))
        for k in keys:
            data.pop(k, None)
        deploy_pred = deploy_mod(data)
        deploy_pred = {k: deploy_pred[k].to("cpu") for k in keys}
        for k in keys:
            assert torch.allclose(train_pred[k], deploy_pred[k], atol=atol)

        # now test info
        # hack for old version
        if sys.version_info[1] > 6:
            text = {"text": True}
        else:
            text = {}
        retcode = subprocess.run(
            ["nequip-deploy", "info", str(deployed_path)],
            stdout=subprocess.PIPE,
            **text,
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
            set_global_options=False,
        )
        # use .get() so it's not transformed
        atoms = dataset.get(dataset.indices()[data_idx]).to_ase()
        atoms.calc = calc
        ase_pred = {
            AtomicDataDict.TOTAL_ENERGY_KEY: atoms.get_potential_energy(),
            AtomicDataDict.FORCE_KEY: atoms.get_forces(),
            AtomicDataDict.PER_ATOM_ENERGY_KEY: atoms.get_potential_energies(),
        }
        assert ase_pred[AtomicDataDict.TOTAL_ENERGY_KEY].shape == tuple()
        assert ase_pred[AtomicDataDict.FORCE_KEY].shape == (len(atoms), 3)
        assert ase_pred[AtomicDataDict.PER_ATOM_ENERGY_KEY].shape == (len(atoms),)
        for k in keys:
            assert torch.allclose(
                deploy_pred[k].squeeze(-1),
                torch.as_tensor(ase_pred[k], dtype=torch.get_default_dtype()),
                atol=atol,
            )
