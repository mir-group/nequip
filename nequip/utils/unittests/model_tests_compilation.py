# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""
Model compilation test mixin.

This mixin provides tests for model compilation workflows including:
- nequip-compile (TorchScript and AOTInductor modes)
- Train-time compilation (torch.compile)
"""

import pytest
import copy
import torch
import pathlib
import subprocess
import os
import uuid
import numpy as np

from nequip.data import AtomicDataDict
from nequip.utils.test import override_irreps_debug
from nequip.utils.versions import _TORCH_GE_2_6, _TORCH_GE_2_10
from nequip.ase import NequIPCalculator

from .utils import _check_and_print
from .model_tests_basic import EnergyModelTestsMixin


class CompilationTestsMixin(EnergyModelTestsMixin):
    """
    Model compilation tests for TorchScript, AOTInductor, and train-time compilation.

    Inherits from EnergyModelTestsMixin (which inherits from BasicModelTestsMixin).
    These tests verify that energy models can be compiled and produce identical outputs
    to the original model.
    """

    @pytest.fixture(scope="class")
    def nequip_compile_tol(self, model_dtype):
        """May be overriden by subclasses.

        Returns tolerance based on ``model_dtype``.
        """
        return {"float32": 5e-5, "float64": 1e-12}[model_dtype]

    @pytest.fixture(scope="class", params=[None])
    def nequip_compile_acceleration_modifiers(self, request):
        """Implemented by subclasses.

        Returns a callable that handles model modification and constraints for nequip-compile tests.
        The callable signature is: (mode, device, model_dtype) -> modifiers_list

        Args:
            mode: compilation mode ("torchscript" or "aotinductor")
            device: target device ("cpu" or "cuda")
            model_dtype: "float32" or "float64"

        Returns:
            modifiers_list: list of modifier names to pass to nequip-compile --modifiers
                           or calls pytest.skip() to skip the test

        Default is None (no modifiers applied).
        """
        return request.param

    @pytest.fixture(scope="class", params=[None])
    def train_time_compile_acceleration_modifiers(self, request):
        """Implemented by subclasses.

        Returns a callable that handles model modification and constraints for train-time compile tests.
        The callable signature is: (device) -> modifiers_list

        Args:
            device: target device ("cpu" or "cuda")

        Returns:
            modifiers_list: list of modifier dicts to apply via nequip.model.modify
                           or calls pytest.skip() to skip the test

        Default is None (no modifiers applied).
        """
        return request.param

    @pytest.mark.parametrize(
        "mode",
        ([] if _TORCH_GE_2_10 else ["torchscript"])
        + (["aotinductor"] if _TORCH_GE_2_6 else []),
    )
    @override_irreps_debug(False)
    def test_nequip_compile(
        self,
        fake_model_training_session,
        device,
        mode,
        nequip_compile_tol,
        nequip_compile_acceleration_modifiers,
    ):
        """Tests `nequip-compile` workflows.

        Covers TorchScript and AOTInductor (ASE target).
        """
        config, tmpdir, env, model_dtype, model_source, structures = (
            fake_model_training_session
        )
        assert torch.get_default_dtype() == torch.float64

        # handle acceleration modifiers
        compile_modifiers = []
        if nequip_compile_acceleration_modifiers is not None:
            compile_modifiers = nequip_compile_acceleration_modifiers(
                mode, device, model_dtype
            )

        # === test nequip-compile ===
        # use checkpoint or package based on fixture
        if model_source == "checkpoint":
            model_path = str(pathlib.Path(f"{tmpdir}/best.ckpt"))
        else:  # package
            model_path = str(pathlib.Path(f"{tmpdir}/orig_package_model.nequip.zip"))

        uid = uuid.uuid4()
        compile_fname = (
            f"compile_model_{uid}.nequip.pt2"
            if mode == "aotinductor"
            else f"compile_model_{uid}.nequip.pth"
        )
        output_path = str(pathlib.Path(f"{tmpdir}/{compile_fname}"))

        # build command with optional modifiers
        cmd = [
            "nequip-compile",
            model_path,
            output_path,
            "--mode",
            mode,
            "--device",
            device,
            # target accepted as argument for both modes, but unused for torchscript mode
            "--target",
            "ase",
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

        # == get ase calculator for checkpoint and compiled models ==
        # we only ever load it into the same device that we `nequip-compile`d for
        chemical_species = config.chemical_species
        chemical_species_to_atom_type_map = {s: s for s in chemical_species}

        # load reference calculator based on model source
        ref_calc = NequIPCalculator._from_saved_model(
            model_path,
            device=device,
            chemical_species_to_atom_type_map=chemical_species_to_atom_type_map,
        )
        compile_calc = NequIPCalculator.from_compiled_model(
            output_path,
            device=device,
            chemical_species_to_atom_type_map=chemical_species_to_atom_type_map,
        )

        # == loop over validation data and do checks ==
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
                rtol=nequip_compile_tol,
                atol=nequip_compile_tol,
            )
            np.testing.assert_allclose(
                ckpt_F,
                compile_F,
                rtol=nequip_compile_tol,
                atol=nequip_compile_tol,
            )

    def compare_output_and_gradients(
        self, modelA, modelB, model_test_data, tol, compare_outputs=None
    ):
        # default fields
        if compare_outputs is None:
            compare_outputs = [
                AtomicDataDict.PER_ATOM_ENERGY_KEY,
                AtomicDataDict.TOTAL_ENERGY_KEY,
                AtomicDataDict.FORCE_KEY,
                AtomicDataDict.VIRIAL_KEY,
            ]

        A_out = modelA(model_test_data.copy())
        B_out = modelB(model_test_data.copy())
        for key in compare_outputs:
            if key in A_out and key in B_out:
                torch.testing.assert_close(A_out[key], B_out[key], atol=tol, rtol=tol)

        # test backwards pass if there are trainable weights
        if any([p.requires_grad for p in modelB.parameters()]):
            B_loss = B_out[AtomicDataDict.TOTAL_ENERGY_KEY].square().sum()
            B_loss.backward()

            A_loss = A_out[AtomicDataDict.TOTAL_ENERGY_KEY].square().sum()
            A_loss.backward()
            compile_params = dict(modelB.named_parameters())
            for k, v in modelA.named_parameters():
                err = torch.max(torch.abs(v.grad - compile_params[k].grad))
                torch.testing.assert_close(
                    v.grad,
                    compile_params[k].grad,
                    atol=tol,
                    rtol=tol,
                    msg=f"failed for {k}, with MaxAbsErr of {err:.6f}",
                )

    @override_irreps_debug(False)
    def test_train_time_compile(
        self, model, model_test_data, device, train_time_compile_acceleration_modifiers
    ):
        """
        Test train-time compilation, i.e. `make_fx` -> `export` -> `AOTAutograd` correctness.
        """
        if not _TORCH_GE_2_6:
            pytest.skip("PT2 compile tests skipped for torch < 2.6")

        # TODO: have better way to test since problem might be version dependent now
        # see https://github.com/pytorch/pytorch/issues/146390
        # i.e. some ops have problems in torch 2.6, but may be ok in torch 2.7
        if device == "cpu":
            pytest.skip(
                "compile tests are skipped for CPU as there are known compilation bugs for both NequIP and Allegro models on CPU"
            )

        instance, config, _ = model
        # get tolerance based on model_dtype
        tol = {
            torch.float32: 5e-5,
            torch.float64: 1e-12,
        }[instance.model_dtype]

        # handle acceleration modifiers
        modifier_configs = []
        if train_time_compile_acceleration_modifiers is not None:
            modifier_configs = train_time_compile_acceleration_modifiers(device)

        # make compiled model
        config = copy.deepcopy(config)
        config["compile_mode"] = "compile"

        # apply modifiers if specified
        if modifier_configs:
            config = {
                "_target_": "nequip.model.modify",
                "modifiers": modifier_configs,
                "model": config,
            }

        compile_model = self.make_model(config, device=device)

        self.compare_output_and_gradients(
            modelA=instance,
            modelB=compile_model,
            model_test_data=model_test_data,
            tol=tol,
        )
