from conftest import TrainingInvarianceBaseTest, _check_and_print
import pytest
import numpy as np
import torch
import os
import pathlib
import subprocess
import uuid
from nequip.data import to_ase
from nequip.ase import NequIPCalculator
from hydra.utils import instantiate
from nequip.utils.versions import _TORCH_GE_2_4


@pytest.mark.skipif(not _TORCH_GE_2_4, reason="OpenEquivariance requires torch >= 2.4")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="OEQ requires CUDA")
class TestOEQTrainingInvariance(TrainingInvarianceBaseTest):
    def modify_model_config(self, original_config):
        try:
            import openequivariance  # noqa: F401
        except ImportError:
            pytest.skip("OpenEquivariance not installed")
        new_config = original_config.copy()
        training_module = new_config["training_module"]
        original_model = training_module["model"]
        training_module["model"] = {
            "_target_": "nequip.model.modify",
            "modifiers": [{"modifier": "enable_OpenEquivariance"}],
            "model": original_model,
        }
        return new_config

    def map_location(self):
        return "cuda"


@pytest.mark.skipif(not _TORCH_GE_2_4, reason="OpenEquivariance requires torch >= 2.4")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="OEQ requires CUDA")
def test_oeq_package_compile_workflow(fake_model_training_session):
    """
    Test the full OEQ workflow: checkpoint -> package -> compile (both paths) -> ASE Calculator.

    This test:
    1. Uses a previously trained model checkpoint
    2. Creates a package file from the checkpoint
    3. Compiles both checkpoint and package to torchscript with OEQ
    4. Creates three NequIPCalculator instances and verifies they give identical results
    """
    try:
        import openequivariance  # noqa: F401
    except ImportError:
        pytest.skip("OpenEquivariance not installed")

    config, tmpdir, env, model_dtype = fake_model_training_session
    device = "cuda"  # OEQ only works on CUDA

    # Set tolerance based on model dtype
    tol = {"float32": 1e-5, "float64": 1e-10}[model_dtype]

    # Path to the checkpoint file from training session
    ckpt_path = str(pathlib.Path(f"{tmpdir}/best.ckpt"))

    # === Step 1: Create package from checkpoint ===
    uid = uuid.uuid4()
    package_path = str(pathlib.Path(f"{tmpdir}/oeq_test_model_{uid}.nequip.zip"))

    retcode = subprocess.run(
        [
            "nequip-package",
            "build",
            ckpt_path,
            package_path,
        ],
        cwd=tmpdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _check_and_print(retcode)
    assert os.path.exists(
        package_path
    ), f"Package file `{package_path}` does not exist!"

    # === Step 2: Compile checkpoint to torchscript with OEQ ===
    compiled_from_ckpt_path = str(
        pathlib.Path(f"{tmpdir}/oeq_compiled_from_ckpt_{uid}.nequip.pth")
    )

    retcode = subprocess.run(
        [
            "nequip-compile",
            ckpt_path,
            compiled_from_ckpt_path,
            "--mode",
            "torchscript",
            "--device",
            device,
            "--target",
            "ase",
            "--modifiers",
            "enable_OpenEquivariance",
        ],
        cwd=tmpdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _check_and_print(retcode)
    assert os.path.exists(
        compiled_from_ckpt_path
    ), f"Compiled model from checkpoint `{compiled_from_ckpt_path}` does not exist!"

    # === Step 3: Compile package to torchscript with OEQ ===
    compiled_from_pkg_path = str(
        pathlib.Path(f"{tmpdir}/oeq_compiled_from_pkg_{uid}.nequip.pth")
    )

    retcode = subprocess.run(
        [
            "nequip-compile",
            package_path,
            compiled_from_pkg_path,
            "--mode",
            "torchscript",
            "--device",
            device,
            "--target",
            "ase",
            "--modifiers",
            "enable_OpenEquivariance",
        ],
        cwd=tmpdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _check_and_print(retcode)
    assert os.path.exists(
        compiled_from_pkg_path
    ), f"Compiled model from package `{compiled_from_pkg_path}` does not exist!"

    # === Step 4: Create ASE calculators (all on CUDA) ===
    # Reference calculator from checkpoint (with OEQ modifiers)
    ref_calc = NequIPCalculator.from_checkpoint_model(
        ckpt_path,
        device=device,
        modifiers=["enable_OpenEquivariance"],
    )

    # Calculator from compiled checkpoint
    compiled_ckpt_calc = NequIPCalculator.from_compiled_model(
        compiled_from_ckpt_path,
        device=device,
    )

    # Calculator from compiled package
    compiled_pkg_calc = NequIPCalculator.from_compiled_model(
        compiled_from_pkg_path,
        device=device,
    )

    # === Step 5: Test on validation data ===
    datamodule = instantiate(config.data, _recursive_=False)
    datamodule.prepare_data()
    datamodule.setup("validate")
    dloader = datamodule.val_dataloader()[0]

    # Test predictions match across all three calculators
    for data in dloader:
        atoms_list = to_ase(data.copy())
        for atoms in atoms_list:
            # Make copies for each calculator
            ref_atoms = atoms.copy()
            compiled_ckpt_atoms = atoms.copy()
            compiled_pkg_atoms = atoms.copy()

            # Set calculators
            ref_atoms.calc = ref_calc
            compiled_ckpt_atoms.calc = compiled_ckpt_calc
            compiled_pkg_atoms.calc = compiled_pkg_calc

            # Get energies and forces
            ref_E = ref_atoms.get_potential_energy()
            ref_F = ref_atoms.get_forces()

            compiled_ckpt_E = compiled_ckpt_atoms.get_potential_energy()
            compiled_ckpt_F = compiled_ckpt_atoms.get_forces()

            compiled_pkg_E = compiled_pkg_atoms.get_potential_energy()
            compiled_pkg_F = compiled_pkg_atoms.get_forces()

            # Clean up atoms objects
            del atoms, ref_atoms, compiled_ckpt_atoms, compiled_pkg_atoms

            # Test energies match
            assert np.allclose(
                ref_E, compiled_ckpt_E, rtol=tol, atol=tol
            ), f"Energy mismatch between reference and compiled from checkpoint: {np.abs(ref_E - compiled_ckpt_E)}"
            assert np.allclose(
                ref_E, compiled_pkg_E, rtol=tol, atol=tol
            ), f"Energy mismatch between reference and compiled from package: {np.abs(ref_E - compiled_pkg_E)}"

            # Test forces match
            assert np.allclose(
                ref_F, compiled_ckpt_F, rtol=tol, atol=tol
            ), f"Force mismatch between reference and compiled from checkpoint: {np.max(np.abs(ref_F - compiled_ckpt_F))}"
            assert np.allclose(
                ref_F, compiled_pkg_F, rtol=tol, atol=tol
            ), f"Force mismatch between reference and compiled from package: {np.max(np.abs(ref_F - compiled_pkg_F))}"
