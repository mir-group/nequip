import pytest
import pathlib
import subprocess

import numpy as np
import torch

from nequip.model import ModelFromPackage
from nequip.nn import graph_model
from nequip.data import to_ase
from nequip.ase import NequIPCalculator

from conftest import _check_and_print
from hydra.utils import instantiate


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_package(BENCHMARK_ROOT, fake_model_training_session, device):
    config, tmpdir, env = fake_model_training_session

    # just in case
    assert torch.get_default_dtype() == torch.float64

    # atol on MODEL dtype, since a mostly float32 model still has float32 variation
    atol = {"float32": 5e-4, "float64": 1e-9}[config.training_module.model.model_dtype]

    # === test mode=build ===
    ckpt_path = pathlib.Path(f"{tmpdir}/last.ckpt")
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

    # === load model and check that metadata saved ===
    metadata = ModelFromPackage(package_path).metadata
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
        set_global_options=False,
    )
    package_calc = NequIPCalculator.from_packaged_model(
        package_path,
        device=device,
        set_global_options=False,
    )

    # == get validation data by instantiating datamodules ==
    datamodule = instantiate(config.data, _recursive_=False)
    datamodule.prepare_data()
    datamodule.setup("validate")
    dloader = datamodule.val_dataloader()[0]

    # == loop over data and do checks ==
    for data in dloader:
        atoms_list = to_ase(data.copy())
        for idx, atoms in enumerate(atoms_list):
            atoms.calc = ckpt_calc
            ckpt_pred = {
                "E": atoms.get_potential_energy(),
                "F": atoms.get_forces(),
            }
            atoms.calc = package_calc
            package_pred = {
                "E": atoms.get_potential_energy(),
                "F": atoms.get_forces(),
            }

            E_err = np.max(np.abs((ckpt_pred["E"] - package_pred["E"])))
            assert E_err <= atol
            F_err = np.max(np.abs((ckpt_pred["F"] - package_pred["F"])))
            assert F_err <= atol
