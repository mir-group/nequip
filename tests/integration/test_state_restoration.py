import pytest
import pathlib
import subprocess

import numpy as np
import torch

from nequip.train.lightning import _SOLE_MODEL_KEY
from nequip.model import ModelFromPackage
from nequip.nn import graph_model
from nequip.data import to_ase
from nequip.ase import NequIPCalculator

from ase.io import read

from conftest import _check_and_print
from hydra.utils import instantiate


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_state_restoration(BENCHMARK_ROOT, fake_model_training_session, device):
    """
    This test checks that the model states are correctly restored based on the following rules.
    1. Doing the `val` or `test` run type after `train` will always use the best model checkpoint (which uses the `ModelCheckpoint`'s default naming convention, but we have configured it to be `best.ckpt` for the tests)
    2. If EMA is used, the EMA model is the one that is loaded from checkpoint for `NequIPCalculator`, `nequip-compile`, and `nequip-package`
    """
    config, tmpdir, env = fake_model_training_session

    # just in case
    assert torch.get_default_dtype() == torch.float64

    # atol on MODEL dtype, since a mostly float32 model still has float32 variation
    atol = {"float32": 2e-4, "float64": 1e-8}[config.training_module.model.model_dtype]

    # === test nequip-package ===
    # !! NOTE: we use the `best.ckpt` because val, test metrics were computed with `best.ckpt` in the `test` run stages !!
    ckpt_path = pathlib.Path(f"{tmpdir}/best.ckpt")
    package_path = pathlib.Path(f"{tmpdir}/packaged_model.nequip.zip")
    retcode = subprocess.run(
        [
            "nequip-package",
            "--ckpt-path",
            f"{str(ckpt_path)}",
            "--output-path",
            f"{str(package_path)}",
        ],
        cwd=tmpdir,
    )
    _check_and_print(retcode)
    assert package_path.is_file(), "`nequip-package` didn't create file"

    # === load validation metrics from checkpoint ===
    # this is the `best.ckpt` checkpoint, so the val metrics should match that model predictions
    checkpoint_dict = torch.load(ckpt_path, weights_only=False)
    val_metrics = checkpoint_dict["state_dict"]["val_metrics.0._extra_state"][
        "metrics_values_epoch"
    ]

    # === load model and check that metadata saved ===
    # we only test single model case (we expect the core `nequip` repo to only house single model `LightningModule`s)
    metadata = ModelFromPackage(package_path)[_SOLE_MODEL_KEY].metadata
    assert np.allclose(
        float(metadata[graph_model.R_MAX_KEY]), config.training_module.model.r_max
    )
    assert len(metadata[graph_model.TYPE_NAMES_KEY].split(" ")) == len(
        config.training_module.model.type_names
    )

    # == get ase calculator for checkpoint and packaged models ==
    ckpt_calc = NequIPCalculator.from_checkpoint_model(
        ckpt_path,
        device=device,
    )
    package_calc = NequIPCalculator.from_packaged_model(
        package_path,
        device=device,
    )

    # == get validation data by instantiating datamodules ==
    datamodule = instantiate(config.data, _recursive_=False)
    datamodule.prepare_data()
    datamodule.setup("validate")
    dloader = datamodule.val_dataloader()[0]

    # == loop over data and do checks ==
    E_abs_list = []
    F_abs_list = []
    for data in dloader:
        atoms_list = to_ase(data.copy())
        for idx, atoms in enumerate(atoms_list):
            E_ref = atoms.get_potential_energy()
            F_ref = atoms.get_forces()

            ckpt_atoms, package_atoms = atoms.copy(), atoms.copy()
            ckpt_atoms.calc = ckpt_calc
            ckpt_E = ckpt_atoms.get_potential_energy()
            ckpt_F = ckpt_atoms.get_forces()

            package_atoms.calc = package_calc
            package_E = package_atoms.get_potential_energy()
            package_F = package_atoms.get_forces()

            del atoms, ckpt_atoms, package_atoms
            E_err = np.max(np.abs((ckpt_E - package_E)))
            assert E_err <= atol, E_err
            F_err = np.max(np.abs((ckpt_F - package_F)))
            assert F_err <= atol, F_err

            E_abs_list.append(np.abs(ckpt_E - E_ref))
            F_abs_list.append(np.abs(ckpt_F - F_ref))

    # `minimal.yaml` and `minimal_emt.yaml` use energy and force MAEs
    E_MAE_ckpt = np.mean(E_abs_list)
    F_MAE_ckpt = np.mean(np.concatenate(F_abs_list, 0))
    train_time_val_E_MAE = val_metrics["total_energy_mae"]
    E_err = np.max(np.abs(train_time_val_E_MAE - E_MAE_ckpt))
    assert E_err <= atol, f"train:{train_time_val_E_MAE},  ckpt: {E_MAE_ckpt}"
    train_time_val_F_MAE = val_metrics["forces_mae"]
    F_err = np.max(np.abs(train_time_val_F_MAE - F_MAE_ckpt))
    assert F_err <= atol, f"train:{train_time_val_F_MAE},  ckpt: {F_MAE_ckpt}"

    # get test metrics
    test_atoms = read(f"{tmpdir}/test_dataset0.xyz", ":")
    for atoms in test_atoms:
        E_ref = atoms.get_potential_energy()
        F_ref = atoms.get_forces()

        ckpt_atoms, package_atoms = atoms.copy(), atoms.copy()
        ckpt_atoms.calc = ckpt_calc
        ckpt_E = ckpt_atoms.get_potential_energy()
        ckpt_F = ckpt_atoms.get_forces()

        package_atoms.calc = package_calc
        package_E = package_atoms.get_potential_energy()
        package_F = package_atoms.get_forces()

        del atoms, ckpt_atoms, package_atoms
        E_err = np.max(np.abs((ckpt_E - package_E)))
        assert E_err <= atol, E_err
        F_err = np.max(np.abs((ckpt_F - package_F)))
        assert F_err <= atol, F_err

        E_err = np.max(np.abs((ckpt_E - E_ref)))
        assert E_err <= atol, E_err
        F_err = np.max(np.abs((ckpt_F - F_ref)))
        assert F_err <= atol, F_err
