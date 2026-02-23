# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""
Model ASE integration test mixin for ``nequip-compile`` workflows.

This mixin provides tests for compiled-model ASE integration workflows including:
- ``nequip-compile`` in TorchScript mode (where available)
- ``nequip-compile`` in AOTInductor mode
"""

import pytest
import torch
import numpy as np

from nequip.utils.test import override_irreps_debug
from nequip.utils.versions import _TORCH_GE_2_6, _TORCH_GE_2_10

from .utils import resolve_saved_model_path, run_nequip_compile
from .model_tests_basic import EnergyModelTestsMixin


class ASEIntegrationMixin(EnergyModelTestsMixin):
    """
    ASE integration tests for compiled-model workflows.

    Inherits from EnergyModelTestsMixin (which inherits from BasicModelTestsMixin).
    These tests verify that models loaded via ``from_compiled_model`` produce
    energies and forces consistent with reference models loaded from saved artifacts.
    """

    @pytest.fixture(scope="class")
    def ase_integration_tol(self, model_dtype):
        """May be overridden by subclasses.

        Returns tolerance for ASE integration tests based on ``model_dtype``.
        """
        return {"float32": 5e-5, "float64": 1e-12}[model_dtype]

    @pytest.fixture(scope="class")
    def ase_calculator_cls(self):
        """May be overridden by subclasses.

        Return ASE calculator class used for saved-model and compiled-model loading.
        """
        from nequip.integrations.ase import NequIPCalculator

        return NequIPCalculator

    @pytest.fixture(scope="class")
    def ase_aoti_target(self):
        """May be overridden by subclasses.

        Return ``nequip-compile --target`` value used for AOTI compilation.
        """
        from nequip.scripts._compile_utils import AOTI_ASE_TARGET

        return AOTI_ASE_TARGET

    @pytest.fixture(scope="class", params=[None])
    def ase_compile_modifiers(self, request):
        """Implemented by subclasses.

        Returns a callable that handles model modification and constraints for
        ASE integration compile tests.
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

    @pytest.fixture(
        scope="class",
        params=([] if _TORCH_GE_2_10 else ["torchscript"])
        + (["aotinductor"] if _TORCH_GE_2_6 else []),
    )
    def ase_compiled_model(
        self,
        request,
        fake_model_training_session,
        device,
        ase_compile_modifiers,
        ase_aoti_target,
    ):
        """Compile model once for ASE integration tests and return saved/compiled paths.

        Parametrized by compilation mode (torchscript/aotinductor).

        Returns:
            tuple[str, str, list]: ``(saved_model_path, compiled_model_path, structures)``
        """
        mode = request.param
        _, tmpdir, env, model_dtype, model_source, structures = (
            fake_model_training_session
        )

        compile_modifiers = []
        if ase_compile_modifiers is not None:
            compile_modifiers = ase_compile_modifiers(mode, device, model_dtype)

        model_path = resolve_saved_model_path(tmpdir, model_source)
        output_path = run_nequip_compile(
            model_path=model_path,
            tmpdir=tmpdir,
            env=env,
            mode=mode,
            device=device,
            target=ase_aoti_target,
            modifiers=compile_modifiers,
            output_prefix="ase_model",
        )

        return model_path, output_path, structures

    @override_irreps_debug(False)
    def test_ase_integration_from_nequip_compile(
        self,
        device,
        ase_integration_tol,
        ase_calculator_cls,
        ase_compiled_model,
    ):
        """Test compiled-model ASE integration from ``nequip-compile`` artifacts."""
        model_path, output_path, structures = ase_compiled_model
        assert torch.get_default_dtype() == torch.float64
        ref_calc = ase_calculator_cls._from_saved_model(
            model_path,
            device=device,
            chemical_species_to_atom_type_map=True,
        )
        compile_calc = ase_calculator_cls.from_compiled_model(
            output_path,
            device=device,
            chemical_species_to_atom_type_map=True,
        )

        for atoms in structures:
            ckpt_atoms, compile_atoms = atoms.copy(), atoms.copy()
            ckpt_atoms.calc = ref_calc
            ckpt_E = ckpt_atoms.get_potential_energy()
            ckpt_F = ckpt_atoms.get_forces()

            compile_atoms.calc = compile_calc
            compile_E = compile_atoms.get_potential_energy()
            compile_F = compile_atoms.get_forces()

            del atoms, ckpt_atoms, compile_atoms
            np.testing.assert_allclose(
                ckpt_E,
                compile_E,
                rtol=ase_integration_tol,
                atol=ase_integration_tol,
            )
            np.testing.assert_allclose(
                ckpt_F,
                compile_F,
                rtol=ase_integration_tol,
                atol=ase_integration_tol,
            )
