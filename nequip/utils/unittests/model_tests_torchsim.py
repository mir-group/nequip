# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""
TorchSim integration test mixin.

This mixin provides tests for torch-sim integration, verifying that NequIPTorchSimCalc
(torch-sim interface) produces identical results to NequIPCalculator (ASE interface).
Tests include model output validation, calculator consistency, and batched evaluation.
"""

import pytest
import torch
import numpy as np
from ase.calculators.calculator import all_properties as ase_all_properties
from ase.calculators.singlepoint import SinglePointCalculator

from nequip.data import AtomicDataDict, from_ase
from nequip.utils.versions import _TORCH_GE_2_6, _TORCH_GE_2_10
from nequip.integrations.ase import NequIPCalculator

from .utils import resolve_saved_model_path, run_nequip_compile
from .model_tests_basic import EnergyModelTestsMixin


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


class TorchSimIntegrationMixin(EnergyModelTestsMixin):
    """
    TorchSim integration tests.

    Inherits from EnergyModelTestsMixin, adding torch-sim-specific integration tests.
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

    @pytest.fixture(scope="class")
    def torchsim_aoti_target(self):
        """May be overridden by subclasses.

        Return ``nequip-compile --target`` value used for AOTI compilation.
        """
        from nequip.scripts._compile_utils import AOTI_BATCH_TARGET

        return AOTI_BATCH_TARGET

    @pytest.fixture(scope="class")
    def torchsim_calculator_cls(self):
        """May be overridden by subclasses.

        Return torch-sim calculator class used for compiled-model loading.
        """
        return NequIPTorchSimCalc

    @pytest.fixture(scope="class")
    def torchsim_reference_ase_calculator_cls(self):
        """May be overridden by subclasses.

        Return ASE calculator class used for saved-model reference evaluation.
        """
        return NequIPCalculator

    @pytest.fixture(scope="class")
    def torchsim_properties_to_compare(self):
        """May be overridden by subclasses.

        Return property names to compare between ASE and torch-sim outputs.

        NOTE: assume ASE property keys and torch-sim output keys are identical.
        """
        return ["energy", "forces", "stress"]

    def _make_ase_reference_data(self, atoms, properties_to_compare):
        """Evaluate ASE properties and convert them into AtomicDataDict format."""
        evaluated = {}
        for property_name in properties_to_compare:
            evaluated[property_name] = atoms.calc.get_property(property_name, atoms)

        sp_atoms = atoms.copy()
        # ASE standard properties go through SinglePointCalculator.
        # Non-standard extension properties are stored in info/arrays and
        # loaded via from_ase(..., include_keys=...).
        calc_results = {
            k: v for k, v in evaluated.items() if k in set(ase_all_properties)
        }
        if calc_results:
            sp_atoms.calc = SinglePointCalculator(sp_atoms, **calc_results)

        for key, value in evaluated.items():
            if key in calc_results:
                continue
            value_np = np.asarray(value)
            if value_np.ndim >= 1 and value_np.shape[0] == len(sp_atoms):
                sp_atoms.arrays[key] = value_np
            else:
                sp_atoms.info[key] = value_np

        return from_ase(sp_atoms, include_keys=list(properties_to_compare))

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
        torchsim_compile_modifiers,
        torchsim_aoti_target,
    ):
        """Compile model once for torch-sim and reuse across tests.

        Parametrized by compilation mode (torchscript/aotinductor).

        Returns:
            tuple[str, str, list]: ``(saved_model_path, compiled_model_path, structures)``
        """
        mode = request.param
        _, tmpdir, env, model_dtype, model_source, structures = (
            fake_model_training_session
        )

        compile_modifiers = []
        if torchsim_compile_modifiers is not None:
            compile_modifiers = torchsim_compile_modifiers(mode, device, model_dtype)

        model_path = resolve_saved_model_path(tmpdir, model_source)
        output_path = run_nequip_compile(
            model_path=model_path,
            tmpdir=tmpdir,
            env=env,
            mode=mode,
            device=device,
            target=torchsim_aoti_target,
            modifiers=compile_modifiers,
            output_prefix="torchsim_model",
        )

        return model_path, output_path, structures

    @pytest.fixture(scope="class", params=[None])
    def torchsim_compile_modifiers(self, request):
        """Implemented by subclasses.

        Returns a callable that handles model modification and constraints for
        torch-sim compile tests.
        The callable signature is: ``(mode, device, model_dtype) -> modifiers_list``

        Args:
            mode: compilation mode (``"torchscript"`` or ``"aotinductor"``)
            device: target device (``"cpu"`` or ``"cuda"``)
            model_dtype: model dtype string (``"float32"`` or ``"float64"``)

        Returns:
            modifiers_list: list of modifier names to pass to
                ``nequip-compile --modifiers`` or calls ``pytest.skip()`` to skip
                the test.

        Default is ``None`` (no modifiers applied).
        """
        return request.param

    @pytest.mark.skipif(not _TORCHSIM_INSTALLED, reason="torch-sim not installed")
    def test_torchsim_model_interface_validation(
        self,
        torchsim_compiled_model,
        device,
        torchsim_tol,
        torchsim_calculator_cls,
    ):
        """Test that NequIPTorchSimCalc follows the torch-sim ModelInterface contract.

        Uses torch-sim's validation utility to check:
        - Model has required attributes (dtype, device, compute_stress, compute_forces)
        - Model outputs have correct shapes
        - Model doesn't mutate inputs
        - Batched evaluation matches individual evaluations
        """
        _, compiled_path, structures = torchsim_compiled_model
        torchsim_calc = torchsim_calculator_cls.from_compiled_model(
            compiled_path, device=device, chemical_species_to_atom_type_map=True
        )
        validate_model_interface_contract(
            torchsim_calc, structures, device, torchsim_tol
        )

    @pytest.mark.skipif(not _TORCHSIM_INSTALLED, reason="torch-sim not installed")
    def test_torchsim_calculator_consistency(
        self,
        torchsim_compiled_model,
        device,
        torchsim_tol,
        torchsim_calculator_cls,
        torchsim_reference_ase_calculator_cls,
        torchsim_properties_to_compare,
    ):
        """Test that NequIPTorchSimCalc matches NequIPCalculator output.

        Compares energies, forces, and stress between torch-sim interface and
        ASE calculator interface. ASE calculator loads from saved model (checkpoint/package)
        as reference, while TorchSim loads from batch-compiled model.
        """
        model_path, compiled_path, structures = torchsim_compiled_model

        # load both calculators
        # ASE calculator from saved model (reference)
        nequip_calc = torchsim_reference_ase_calculator_cls._from_saved_model(
            model_path,
            device=device,
            chemical_species_to_atom_type_map=True,
        )

        # TorchSim calculator from batch-compiled model
        torchsim_calc = torchsim_calculator_cls.from_compiled_model(
            compiled_path,
            device=device,
            chemical_species_to_atom_type_map=True,
        )

        # test on validation structures
        for atoms in structures:
            nequip_atoms = atoms.copy()
            nequip_atoms.calc = nequip_calc

            # TorchSim calculator results
            sim_state = ts.io.atoms_to_state(
                [atoms], device=device, dtype=torch.float64
            )
            ts_results = torchsim_calc(sim_state)
            ase_ref = self._make_ase_reference_data(
                nequip_atoms, torchsim_properties_to_compare
            )

            for property_name in torchsim_properties_to_compare:
                if property_name not in ts_results:
                    continue
                atomic_key = (
                    AtomicDataDict.TOTAL_ENERGY_KEY
                    if property_name == "energy"
                    else property_name
                )
                if atomic_key not in ase_ref:
                    continue

                ase_val = ase_ref[atomic_key].detach().cpu().numpy()
                ts_val = np.reshape(
                    ts_results[property_name].detach().cpu().numpy(), ase_val.shape
                )

                np.testing.assert_allclose(
                    ase_val,
                    ts_val,
                    rtol=torchsim_tol,
                    atol=torchsim_tol,
                    err_msg=f"Mismatch for torch-sim property `{property_name}`",
                )

            del nequip_atoms, sim_state, ts_results

    @pytest.mark.skipif(not _TORCHSIM_INSTALLED, reason="torch-sim not installed")
    @pytest.mark.parametrize("batch_size", [2, 3])
    def test_torchsim_batched_evaluation(
        self,
        torchsim_compiled_model,
        device,
        batch_size,
        torchsim_tol,
        torchsim_calculator_cls,
    ):
        """Test batched evaluation consistency.

        Verifies that evaluating multiple systems in a single batched forward pass
        produces identical results to evaluating each system individually.

        This is a key feature of torch-sim for efficient MD simulations.
        """
        _, compiled_path, structures = torchsim_compiled_model

        # load calculator
        torchsim_calc = torchsim_calculator_cls.from_compiled_model(
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
