import pytest

import numpy as np
import torch

from nequip.data import to_ase
from nequip.ase import NequIPCalculator
from nequip.utils.versions import _TORCH_GE_2_6
from nequip.utils.test import override_irreps_debug
from conftest import _check_and_print

import os
import pathlib
import subprocess
import uuid
from hydra.utils import instantiate


available_devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "mode", ["torchscript"] + (["aotinductor"] if _TORCH_GE_2_6 else [])
)
@override_irreps_debug(False)
def test_compile(fake_model_training_session, device, mode):

    # TODO: sort out the CPU compilation issues
    if mode == "aotinductor" and device == "cpu":
        pytest.skip(
            "compile tests are skipped for CPU as there are known compilation bugs for both NequIP and Allegro models on CPU"
        )

    config, tmpdir, env, model_dtype = fake_model_training_session

    # just in case
    assert torch.get_default_dtype() == torch.float64

    # atol on MODEL dtype, since a mostly float32 model still has float32 variation
    tol = {"float32": 1e-5, "float64": 1e-10}[model_dtype]

    # === test nequip-package ===
    # !! NOTE: we use the `best.ckpt` because val, test metrics were computed with `best.ckpt` in the `test` run stages !!
    ckpt_path = str(pathlib.Path(f"{tmpdir}/best.ckpt"))

    uid = uuid.uuid4()
    compile_fname = (
        f"compile_model_{uid}.nequip.pt2"
        if mode == "aotinductor"
        else f"compile_model_{uid}.nequip.pth"
    )
    output_path = str(pathlib.Path(f"{tmpdir}/{compile_fname}"))
    retcode = subprocess.run(
        [
            "nequip-compile",
            ckpt_path,
            output_path,
            "--mode",
            mode,
            "--device",
            device,
            # target accepted as argument for both modes, but unused for torchscript mode
            "--target",
            "ase",
        ],
        cwd=tmpdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _check_and_print(retcode)
    assert os.path.exists(
        output_path
    ), f"Compiled model `{output_path}` does not exist!"

    # == get ase calculator for checkpoint and compiled models ==
    for load_device in available_devices:
        if mode == "aotinductor" and load_device != device:
            with pytest.raises(
                RuntimeError
            ):  # aotinductor does not support loading on different devices
                NequIPCalculator.from_compiled_model(
                    output_path,
                    device=load_device,
                )
            continue

        # test both devices if possible, where we are testing swapping of devices between compile and
        # loading (CUDA -> cpu and vice versa)
        chemical_symbols = config.chemical_symbols
        ckpt_calc = NequIPCalculator._from_checkpoint_model(
            ckpt_path,
            device=load_device,
            chemical_symbols=chemical_symbols,
        )
        compile_calc = NequIPCalculator.from_compiled_model(
            output_path,
            device=load_device,
            chemical_symbols=chemical_symbols,
        )

        # == get validation data by instantiating datamodules ==
        datamodule = instantiate(config.data, _recursive_=False)
        datamodule.prepare_data()
        datamodule.setup("validate")
        dloader = datamodule.val_dataloader()[0]

        # == loop over data and do checks ==
        for data in dloader:
            atoms_list = to_ase(data.copy())
            for atoms in atoms_list:
                ckpt_atoms, compile_atoms = atoms.copy(), atoms.copy()
                ckpt_atoms.calc = ckpt_calc
                ckpt_E = ckpt_atoms.get_potential_energy()
                ckpt_F = ckpt_atoms.get_forces()

                compile_atoms.calc = compile_calc
                compile_E = compile_atoms.get_potential_energy()
                compile_F = compile_atoms.get_forces()

                del atoms, ckpt_atoms, compile_atoms
                assert np.allclose(ckpt_E, compile_E, rtol=tol, atol=tol), np.max(
                    np.abs((ckpt_E - compile_E))
                )
                assert np.allclose(ckpt_E, compile_E, rtol=tol, atol=tol), np.max(
                    np.abs((ckpt_F - compile_F))
                )
