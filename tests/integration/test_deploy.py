import pytest
import pathlib
import subprocess

import math
import numpy as np
import torch

import nequip
from nequip.train import NequIPLightningModule
from nequip.data import AtomicDataDict, to_ase
import nequip.utils
from nequip.scripts import deploy
from nequip.ase import NequIPCalculator

from conftest import _check_and_print
from hydra.utils import instantiate


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_deploy(BENCHMARK_ROOT, fake_model_training_session, device):
    config, tmpdir, env = fake_model_training_session
    dtype = nequip.utils.dtype_to_name(torch.get_default_dtype())

    # atol on MODEL dtype, since a mostly float32 model still has float32 variation
    atol = {"float32": 5e-4, "float64": 1e-10}[config.training_module.model.model_dtype]

    # === test mode=build ===
    deployed_path = pathlib.Path(f"deployed_{dtype}.pth")
    retcode = subprocess.run(
        [
            "nequip-deploy",
            "build",
            "-ckpt_path",
            f"{tmpdir}/last.ckpt",
            "-out_file",
            f"{str(deployed_path)}",
        ],
        cwd=tmpdir,
    )
    _check_and_print(retcode)
    deployed_path = tmpdir / deployed_path
    assert deployed_path.is_file(), "Deploy didn't create file"

    # === test mode=info ===
    text = {"text": True}
    retcode = subprocess.run(
        [
            "nequip-deploy",
            "info",
            f"{str(deployed_path)}",
        ],
        cwd=tmpdir,
        stdout=subprocess.PIPE,
        **text,
    )
    _check_and_print(retcode)

    # === load model and check that metadata saved ===
    deploy_mod, metadata = deploy.load_deployed_model(
        deployed_path,
        device=device,
        set_global_options=False,  # don't need this corrupting test environment
    )
    # Everything we store right now is ASCII, so decode for printing
    assert metadata[deploy.NEQUIP_VERSION_KEY] == nequip.__version__
    assert np.allclose(
        float(metadata[deploy.R_MAX_KEY]), config.training_module.model.r_max
    )
    assert len(metadata[deploy.TYPE_NAMES_KEY].split(" ")) == len(
        config.training_module.model.type_names
    )

    # Two checks are done in one go in the following
    # 1. check that validation metrics from checkpoint match validation metrics calculated with deployed model
    # 2. check that NequIP calculator gives same outputs as the deployed model

    # == get necessary info and tools from checkpoint ==
    # load NequIPLightningModule from checkpoint
    nequip_module = NequIPLightningModule.load_from_checkpoint(f"{tmpdir}/last.ckpt")
    # extract state of validation metrics from last epoch
    ckpt_val_metrics = nequip_module.val_metrics[0].metrics_values_epoch
    # use the MetricsManager on the deployed model
    val_metrics_manager = nequip_module.val_metrics[0]
    val_metrics_manager.reset()

    # == get ase.calculator ==
    calc = NequIPCalculator.from_deployed_model(
        deployed_path,
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
        data = AtomicDataDict.to_(data, device)
        atoms_list = to_ase(data.copy())
        # run through deployed model
        out_data = deploy_mod(data.copy())
        _ = val_metrics_manager(out_data, data)

        # run through NequIP's ase calculator
        for idx, atoms in enumerate(atoms_list):
            atoms.calc = calc
            ase_pred = {
                AtomicDataDict.TOTAL_ENERGY_KEY: atoms.get_potential_energy(),
                AtomicDataDict.FORCE_KEY: atoms.get_forces(),
            }

            from_deployed = (
                AtomicDataDict.frame_from_batched(out_data.copy(), idx)[
                    AtomicDataDict.TOTAL_ENERGY_KEY
                ]
                .reshape(-1)
                .detach()
            )
            from_ase = torch.as_tensor(
                ase_pred[AtomicDataDict.TOTAL_ENERGY_KEY],
                dtype=torch.get_default_dtype(),
                device=device,
            )
            err = (from_deployed - from_ase).abs().max()
            assert err <= atol

            from_deployed = AtomicDataDict.frame_from_batched(out_data.copy(), idx)[
                AtomicDataDict.FORCE_KEY
            ].detach()
            from_ase = torch.as_tensor(
                ase_pred[AtomicDataDict.FORCE_KEY],
                dtype=torch.get_default_dtype(),
                device=device,
            )
            err = (from_deployed - from_ase).abs().max()
            assert err <= atol

    # == check validation epoch statistics wrt checkpoint ==
    _ = val_metrics_manager.compute()
    deployed_val_metrics = val_metrics_manager.metrics_values_epoch
    assert len(ckpt_val_metrics) == len(deployed_val_metrics)
    assert all(
        [
            math.isclose(a, b, abs_tol=atol)
            for a, b in zip(ckpt_val_metrics, deployed_val_metrics)
        ]
    )
