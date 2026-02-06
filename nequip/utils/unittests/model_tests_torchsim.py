# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""
TorchSim integration test mixin.

This mixin provides tests for torch-sim integration, verifying that NequIPTorchSimCalc
(torch-sim interface) produces identical results to NequIPCalculator (ASE interface).
Tests include model output validation, calculator consistency, and batched evaluation.
"""

import pytest
import torch
import pathlib
import subprocess
import os
import uuid
import numpy as np

from nequip.utils.versions import _TORCH_GE_2_6, _TORCH_GE_2_10
from nequip.ase import NequIPCalculator

from .utils import _check_and_print
from .model_tests_compilation import CompilationTestsMixin


# === TorchSim availability check ===
try:
    import torch_sim as ts
    from nequip.integrations.torchsim import NequIPTorchSimCalc

    _TORCHSIM_INSTALLED = True
except ImportError:
    _TORCHSIM_INSTALLED = False


# === TorchSim test utilities (adapted from torch-sim) ===


def validate_model_interface_contract(
    model, test_structures, device, tol, dtype=torch.float64
):
    """Validate that a model follows the torch-sim ModelInterface contract.

    Adapted from torch-sim's make_validate_model_outputs_test.

    Args:
        model: Model to validate (should implement ModelInterface)
        test_structures: List of ASE Atoms objects to use for testing (at least 2)
        device: Device to run on
        tol: Tolerance for numerical comparisons
        dtype: Data type for test structures

    Raises:
        AssertionError: If model violates the interface contract
    """
    assert model.dtype is not None
    assert model.device is not None
    assert model.compute_stress is not None
    assert model.compute_forces is not None

    try:
        if not model.compute_stress:
            model.compute_stress = True
        stress_computed = True
    except NotImplementedError:
        stress_computed = False

    try:
        if not model.compute_forces:
            model.compute_forces = True
        force_computed = True
    except NotImplementedError:
        force_computed = False

    # use all provided test structures
    n_systems = len(test_structures)
    n_atoms_total = sum(len(s) for s in test_structures)

    # test batched evaluation
    sim_state = ts.io.atoms_to_state(test_structures, device, dtype)

    # store original state to check for mutation
    og_positions = sim_state.positions.clone()
    og_cell = sim_state.cell.clone()
    og_batch = sim_state.system_idx.clone()
    og_atomic_nums = sim_state.atomic_numbers.clone()

    # run model
    model_output = model.forward(sim_state)

    # assert model did not mutate the input
    assert torch.allclose(og_positions, sim_state.positions), "Model mutated positions"
    assert torch.allclose(og_cell, sim_state.cell), "Model mutated cell"
    assert torch.allclose(og_batch, sim_state.system_idx), "Model mutated batch indices"
    assert torch.allclose(og_atomic_nums, sim_state.atomic_numbers), (
        "Model mutated atomic numbers"
    )

    # assert model output has the correct keys
    assert "energy" in model_output, "Model output missing 'energy' key"
    if force_computed:
        assert "forces" in model_output, "Model output missing 'forces' key"
    if stress_computed:
        assert "stress" in model_output, "Model output missing 'stress' key"

    # assert model output shapes are correct
    assert model_output["energy"].shape == (n_systems,), (
        f"Energy shape mismatch: {model_output['energy'].shape}"
    )
    if force_computed:
        assert model_output["forces"].shape == (
            n_atoms_total,
            3,
        ), f"Forces shape mismatch: {model_output['forces'].shape}"
    if stress_computed:
        assert model_output["stress"].shape == (
            n_systems,
            3,
            3,
        ), f"Stress shape mismatch: {model_output['stress'].shape}"

    # test individual evaluations match batched
    atom_offset = 0
    for i, struct in enumerate(test_structures):
        state = ts.io.atoms_to_state([struct], device, dtype)
        output = model.forward(state)

        # check energy consistency
        torch.testing.assert_close(
            output["energy"][0],
            model_output["energy"][i],
            atol=tol,
            rtol=tol,
        )

        # check forces consistency
        if force_computed:
            torch.testing.assert_close(
                output["forces"],
                model_output["forces"][atom_offset : atom_offset + len(struct)],
                atol=tol,
                rtol=tol,
            )

        # check single system output shapes
        assert output["energy"].shape == (1,)
        if force_computed:
            assert output["forces"].shape == (len(struct), 3)
        if stress_computed:
            assert output["stress"].shape == (1, 3, 3)

        atom_offset += len(struct)


class TorchSimIntegrationMixin(CompilationTestsMixin):
    """
    TorchSim integration tests.

    Inherits from CompilationTestsMixin, adding torch-sim-specific integration tests.
    Tests that NequIPTorchSimCalc (compiled with --target batch) produces results matching
    NequIPCalculator (reference ASE interface).

    Includes:
    - Model output validation (using torch-sim test utilities)
    - Calculator consistency tests (using torch-sim test utilities)
    - Batched evaluation tests (custom implementation)
    """

    @pytest.fixture(scope="class")
    def torchsim_tol(self, model_dtype):
        """May be overriden by subclasses.

        Returns tolerance for torch-sim integration tests based on ``model_dtype``.
        """
        return {"float32": 5e-5, "float64": 1e-10}[model_dtype]

    @pytest.fixture(
        scope="class",
        params=([] if _TORCH_GE_2_10 else ["torchscript"])
        + (["aotinductor"] if _TORCH_GE_2_6 else []),
    )
    def torchsim_compiled_model(
        self,
        request,
        fake_model_training_session,
        device,
        nequip_compile_acceleration_modifiers,
    ):
        """Compile model once for torch-sim and reuse across tests.

        Parametrized by compilation mode (torchscript/aotinductor).
        Compiles with --target batch for torch-sim batched evaluation.
        """
        mode = request.param
        config, tmpdir, env, model_dtype, model_source, _ = fake_model_training_session

        # handle acceleration modifiers
        compile_modifiers = []
        if nequip_compile_acceleration_modifiers is not None:
            compile_modifiers = nequip_compile_acceleration_modifiers(
                mode, device, model_dtype
            )

        # get model path
        if model_source == "checkpoint":
            model_path = str(pathlib.Path(f"{tmpdir}/best.ckpt"))
        else:  # package
            model_path = str(pathlib.Path(f"{tmpdir}/orig_package_model.nequip.zip"))

        # compile with --target batch for torch-sim
        uid = uuid.uuid4()
        compile_fname = (
            f"torchsim_model_{uid}.nequip.pt2"
            if mode == "aotinductor"
            else f"torchsim_model_{uid}.nequip.pth"
        )
        output_path = str(pathlib.Path(f"{tmpdir}/{compile_fname}"))

        cmd = [
            "nequip-compile",
            model_path,
            output_path,
            "--mode",
            mode,
            "--device",
            device,
            "--target",
            "batch",  # Key difference from ASE compilation
        ]
        if compile_modifiers:
            cmd.extend(["--modifiers"] + compile_modifiers)

        retcode = subprocess.run(
            cmd,
            cwd=tmpdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _check_and_print(retcode)
        assert os.path.exists(output_path), (
            f"Compiled model `{output_path}` does not exist!"
        )

        return output_path, mode

    @pytest.mark.skipif(not _TORCHSIM_INSTALLED, reason="torch-sim not installed")
    def test_torchsim_model_interface_validation(
        self,
        torchsim_compiled_model,
        device,
        fake_model_training_session,
        torchsim_tol,
    ):
        """Test that NequIPTorchSimCalc follows the torch-sim ModelInterface contract.

        Uses torch-sim's validation utility to check:
        - Model has required attributes (dtype, device, compute_stress, compute_forces)
        - Model outputs have correct shapes
        - Model doesn't mutate inputs
        - Batched evaluation matches individual evaluations
        """
        compiled_path, _ = torchsim_compiled_model
        _, _, _, _, _, structures = fake_model_training_session
        torchsim_calc = NequIPTorchSimCalc.from_compiled_model(
            compiled_path, device=device, chemical_species_to_atom_type_map=True
        )
        validate_model_interface_contract(
            torchsim_calc, structures, device, torchsim_tol
        )

    @pytest.mark.skipif(not _TORCHSIM_INSTALLED, reason="torch-sim not installed")
    def test_torchsim_calculator_consistency(
        self,
        torchsim_compiled_model,
        fake_model_training_session,
        device,
        torchsim_tol,
    ):
        """Test that NequIPTorchSimCalc matches NequIPCalculator output.

        Compares energies, forces, and stress between torch-sim interface and
        ASE calculator interface. ASE calculator loads from saved model (checkpoint/package)
        as reference, while TorchSim loads from batch-compiled model.
        """
        compiled_path, mode = torchsim_compiled_model
        config, tmpdir, env, model_dtype, model_source, structures = (
            fake_model_training_session
        )

        # get model path for ASE calculator
        if model_source == "checkpoint":
            model_path = str(pathlib.Path(f"{tmpdir}/best.ckpt"))
        else:  # package
            model_path = str(pathlib.Path(f"{tmpdir}/orig_package_model.nequip.zip"))

        # load both calculators
        # ASE calculator from saved model (reference)
        nequip_calc = NequIPCalculator._from_saved_model(
            model_path,
            device=device,
            chemical_species_to_atom_type_map=True,
        )

        # TorchSim calculator from batch-compiled model
        torchsim_calc = NequIPTorchSimCalc.from_compiled_model(
            compiled_path,
            device=device,
            chemical_species_to_atom_type_map=True,
        )

        # test on validation structures
        for atoms in structures:
            # NequIP calculator results
            nequip_atoms = atoms.copy()
            nequip_atoms.calc = nequip_calc
            nequip_E = nequip_atoms.get_potential_energy()
            nequip_F = nequip_atoms.get_forces()
            nequip_S = nequip_atoms.get_stress(voigt=False)

            # TorchSim calculator results
            # convert atoms to SimState
            sim_state = ts.io.atoms_to_state(
                [atoms], device=device, dtype=torch.float64
            )
            ts_results = torchsim_calc(sim_state)

            # compare energies
            np.testing.assert_allclose(
                nequip_E,
                ts_results["energy"].cpu().numpy()[0],
                rtol=torchsim_tol,
                atol=torchsim_tol,
            )

            # compare forces
            np.testing.assert_allclose(
                nequip_F,
                ts_results["forces"].cpu().numpy(),
                rtol=torchsim_tol,
                atol=torchsim_tol,
            )

            # compare stress (if available)
            if "stress" in ts_results:
                np.testing.assert_allclose(
                    nequip_S,
                    ts_results["stress"].cpu().numpy()[0],
                    rtol=torchsim_tol,
                    atol=torchsim_tol,
                )

            del nequip_atoms, sim_state, ts_results

    @pytest.mark.skipif(not _TORCHSIM_INSTALLED, reason="torch-sim not installed")
    @pytest.mark.parametrize("batch_size", [2, 3])
    def test_torchsim_batched_evaluation(
        self,
        torchsim_compiled_model,
        fake_model_training_session,
        device,
        batch_size,
        torchsim_tol,
    ):
        """Test batched evaluation consistency.

        Verifies that evaluating multiple systems in a single batched forward pass
        produces identical results to evaluating each system individually.

        This is a key feature of torch-sim for efficient MD simulations.
        """
        compiled_path, _ = torchsim_compiled_model
        config, _, _, _, _, structures = fake_model_training_session

        # load calculator
        torchsim_calc = NequIPTorchSimCalc.from_compiled_model(
            compiled_path,
            device=device,
            chemical_species_to_atom_type_map=True,
        )

        # check if we have enough structures for batched evaluation
        if len(structures) < batch_size:
            pytest.skip(f"Not enough structures for batch_size={batch_size}")

        test_structures = structures[:batch_size]

        # === test batched vs individual evaluation ===

        # individual evaluations
        individual_energies = []
        individual_forces = []
        individual_stresses = []

        for atoms in test_structures:
            sim_state = ts.io.atoms_to_state(
                [atoms], device=device, dtype=torch.float64
            )
            result = torchsim_calc(sim_state)
            individual_energies.append(result["energy"].cpu().numpy()[0])
            individual_forces.append(result["forces"].cpu().numpy())
            if "stress" in result:
                individual_stresses.append(result["stress"].cpu().numpy()[0])

        # batched evaluation
        batched_sim_state = ts.io.atoms_to_state(
            test_structures, device=device, dtype=torch.float64
        )
        batched_result = torchsim_calc(batched_sim_state)

        # compare results
        batched_energies = batched_result["energy"].cpu().numpy()
        batched_forces = batched_result["forces"].cpu().numpy()

        # compare energies
        np.testing.assert_allclose(
            np.array(individual_energies),
            batched_energies,
            rtol=torchsim_tol,
            atol=torchsim_tol,
            err_msg="Batched energies don't match individual evaluations",
        )

        # compare forces (concatenated)
        individual_forces_concat = np.concatenate(individual_forces, axis=0)
        np.testing.assert_allclose(
            individual_forces_concat,
            batched_forces,
            rtol=torchsim_tol,
            atol=torchsim_tol,
            err_msg="Batched forces don't match individual evaluations",
        )

        # compare stresses if available
        if "stress" in batched_result and individual_stresses:
            batched_stresses = batched_result["stress"].cpu().numpy()
            np.testing.assert_allclose(
                np.array(individual_stresses),
                batched_stresses,
                rtol=torchsim_tol,
                atol=torchsim_tol,
                err_msg="Batched stresses don't match individual evaluations",
            )
