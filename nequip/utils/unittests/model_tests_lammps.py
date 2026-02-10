# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""
LAMMPS ML-IAP integration test mixin.

This mixin provides tests for LAMMPS ML-IAP integration, verifying that
models compiled for LAMMPS produce identical results to direct NequIP calculations.
"""

import pytest
import torch
import pathlib
import subprocess
import os
import textwrap

import numpy as np
import ase.io

from nequip.data import to_ase
from nequip.ase import NequIPCalculator
from nequip.utils.versions import _TORCH_GE_2_6
from nequip.utils.global_dtype import _GLOBAL_DTYPE

from hydra.utils import instantiate
from .utils import _check_and_print
from .model_tests_basic import EnergyModelTestsMixin


# === LAMMPS availability checks for MLIAP tests ===
LAMMPS = os.environ.get("LAMMPS", "lmp")
LAMMPS_ENV_PREFIX = os.environ.get("LAMMPS_ENV_PREFIX", "")

# Check LAMMPS and required features
HAS_LAMMPS = False
HAS_KOKKOS = False
HAS_KOKKOS_CUDA = False
HAS_MLIAP = False

try:
    _lmp_help = subprocess.run(
        " ".join([LAMMPS_ENV_PREFIX, LAMMPS, "-h"]),
        shell="True",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    HAS_LAMMPS = True
    HAS_KOKKOS = b"mliap/kk" in _lmp_help
    HAS_KOKKOS_CUDA = b"KOKKOS package API: CUDA" in _lmp_help
    HAS_MLIAP = b"ML-IAP" in _lmp_help
except (subprocess.CalledProcessError, FileNotFoundError):
    pass  # Keep defaults as False


class LAMMPSMLIAPIntegrationMixin(EnergyModelTestsMixin):
    """
    LAMMPS ML-IAP integration tests.

    Inherits from EnergyModelTestsMixin, adding LAMMPS ML-IAP-specific integration tests.
    Tests that models compiled for LAMMPS ML-IAP (via nequip-prepare-lmp-mliap)
    produce results matching direct NequIP calculations.

    Note: This is for the ML-IAP interface, not the C++ pair style.
    """

    @pytest.fixture(scope="class")
    def mliap_tol(self, model_dtype):
        """May be overriden by subclasses.

        Returns tolerance for MLIAP integration tests based on ``model_dtype``.
        """
        return {"float32": 1e-5, "float64": 1e-10}[model_dtype]

    @pytest.fixture(scope="class", params=[None])
    def mliap_acceleration_modifiers(self, request):
        """Override in subclasses to specify MLIAP-compatible modifiers.

        Should return a callable with signature: (compile, model_dtype) -> List[str]
        or None to skip modifier application. The callable can call pytest.skip()
        to skip tests based on any of these parameters.
        """
        return request.param

    def _prepare_mliap_model(self, fake_model_training_session, compile, modifiers_str):
        """Prepare MLIAP model from trained model with specified options."""
        config, tmpdir, env, model_dtype, model_source, _ = fake_model_training_session

        # use checkpoint or packaged model path that fixture provides
        if model_source in ("fresh", "checkpoint"):
            model_path = str(pathlib.Path(f"{tmpdir}/best.ckpt"))
        else:  # package
            model_path = str(pathlib.Path(f"{tmpdir}/orig_package_model.nequip.zip"))

        # prepare compile flag and MLIAP file path
        compile_flag = "" if compile else "--no-compile"
        mliap_path = (
            tmpdir
            + f"/model_{model_source}_{compile_flag.replace('--', '').replace('-', '_') or 'compiled'}.nequip.lmp.pt"
        )

        # build nequip-prepare-lmp-mliap command
        cmd = [
            "nequip-prepare-lmp-mliap",
            model_path,
            mliap_path,
        ]
        if modifiers_str:
            cmd.extend(["--modifiers"] + modifiers_str.split())
        if compile_flag:
            cmd.append(compile_flag)

        retcode = subprocess.run(
            cmd,
            cwd=tmpdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _check_and_print(retcode)

        # load calculator for comparison
        chemical_species_to_atom_type_map = {s: s for s in config.chemical_species}
        calc = NequIPCalculator._from_saved_model(
            model_path,
            chemical_species_to_atom_type_map=chemical_species_to_atom_type_map,
        )

        return tmpdir, calc, config, mliap_path

    def _run_lammps_mliap_test(
        self, tmpdir, calc, config, mliap_path, model_dtype, mliap_tol
    ):
        """Run LAMMPS with MLIAP model and compare results with NequIP calculator."""
        # get test structures from datamodule
        torch.set_default_dtype(_GLOBAL_DTYPE)
        datamodule = instantiate(config.data, _recursive_=False)
        datamodule.prepare_data()
        datamodule.setup("test")
        dloader = datamodule.test_dataloader()[0]

        structures = []
        for data in dloader:
            structures += to_ase(data)

        # ensure structures have cells and are wrapped
        if not all(structures[0].pbc):
            L = 50.0
            for struct in structures:
                struct.cell = L * np.eye(3)
                struct.center()
        for s in structures:
            s.wrap()
        structures = structures[:1]  # test with first structure only

        num_types = len(config["chemical_species"])
        periodic = all(structures[0].pbc)
        PRECISION_CONST = 1e6

        # LAMMPS input template
        newline = "\n"
        lmp_in = textwrap.dedent(
            f"""
            units		  metal
            atom_style	  atomic
            newton        on

            thermo 1

            # boundary conditions
            boundary    {"p p p" if periodic else "f f f"}

            # read data (this defines the simulation box)
            read_data structure.data

            # potential
            pair_style mliap unified {mliap_path} 0
            pair_coeff * * {" ".join(sorted(set(config["chemical_species"])))}
            {newline.join([f"mass {i + 1} 1.0" for i in range(num_types)])}

            neighbor	1.0 bin
            neigh_modify    delay 0 every 1 check no

            fix		1 all nve

            timestep	0.001

            compute atomicenergies all pe/atom
            compute totalatomicenergy all reduce sum c_atomicenergies
            compute stress all pressure NULL virial

            thermo_style custom step time temp pe c_totalatomicenergy etotal press spcpu cpuremain c_stress[*]
            run 0
            print "$({PRECISION_CONST} * c_stress[1]) $({PRECISION_CONST} * c_stress[2]) $({PRECISION_CONST} * c_stress[3]) $({PRECISION_CONST} * c_stress[4]) $({PRECISION_CONST} * c_stress[5]) $({PRECISION_CONST} * c_stress[6])" file stress.dat
            print $({PRECISION_CONST} * pe) file pe.dat
            print $({PRECISION_CONST} * c_totalatomicenergy) file totalatomicenergy.dat
            write_dump all custom output.dump id type x y z fx fy fz c_atomicenergies modify format float %20.15g
            """
        ).strip()

        tol = mliap_tol

        for i, structure in enumerate(structures):
            # create LAMMPS run directory
            lammps_tmpdir = tmpdir + f"/lammps_run_{i}"
            os.makedirs(lammps_tmpdir, exist_ok=True)

            # write structure using ASE's LAMMPS data format
            ase.io.write(
                lammps_tmpdir + "/structure.data",
                structure,
                format="lammps-data",
            )

            with open(lammps_tmpdir + "/in.lammps", "w") as f:
                f.write(lmp_in)

            # set environment variables
            env = dict(os.environ)
            env["_NEQUIP_LOG_LEVEL"] = "DEBUG"
            env["CUDA_VISIBLE_DEVICES"] = "0"

            # run LAMMPS with Kokkos
            retcode = subprocess.run(
                [
                    "mpirun",
                    "-np",
                    "1",
                    LAMMPS,
                    "-k",
                    "on",
                    "g",
                    "1",
                    "-sf",
                    "kk",
                    "-pk",
                    "kokkos",
                    "newton",
                    "on",
                    "neigh",
                    "half",
                    "-in",
                    "in.lammps",
                ],
                cwd=lammps_tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            print(retcode.stdout.decode("ascii"))
            _check_and_print(retcode)

            # load LAMMPS results
            lammps_result = ase.io.read(
                lammps_tmpdir + "/output.dump", format="lammps-dump-text"
            )
            structure.calc = calc

            # compare atomic energies
            nequip_atomic_energies = structure.get_potential_energies()
            lammps_atomic_energies = lammps_result.arrays["c_atomicenergies"].reshape(
                -1
            )
            np.testing.assert_allclose(
                nequip_atomic_energies,
                lammps_atomic_energies,
                atol=tol,
                rtol=tol,
            )

            # compare total energy
            lammps_pe = (
                float(pathlib.Path(lammps_tmpdir + "/pe.dat").read_text())
                / PRECISION_CONST
            )
            nequip_pe = structure.get_potential_energy()
            np.testing.assert_allclose(
                lammps_pe,
                nequip_pe,
                atol=tol,
                rtol=tol,
            )

            # compare forces
            np.testing.assert_allclose(
                structure.get_forces(),
                lammps_result.get_forces(),
                atol=tol,
                rtol=tol,
            )

    @pytest.mark.skipif(not HAS_LAMMPS, reason="LAMMPS not available")
    @pytest.mark.skipif(not HAS_MLIAP, reason="LAMMPS not built with ML-IAP")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not HAS_KOKKOS_CUDA, reason="LAMMPS not built with Kokkos CUDA")
    @pytest.mark.skipif(not _TORCH_GE_2_6, reason="MLIAP requires torch >= 2.6")
    @pytest.mark.parametrize("compile", [True, False])
    def test_mliap_integration(
        self,
        fake_model_training_session,
        mliap_acceleration_modifiers,
        model_dtype,
        compile,
        mliap_tol,
    ):
        """Test LAMMPS MLIAP integration matches direct NequIP calculations.

        Sharp edges
         - it is important for structures to have orthorhombic cells
        """
        # handle acceleration modifiers (can skip based on model_dtype, etc.)
        modifiers_str = ""
        if mliap_acceleration_modifiers is not None:
            modifiers_list = mliap_acceleration_modifiers(compile, model_dtype)
            if modifiers_list:
                modifiers_str = " ".join(modifiers_list)

        # prepare MLIAP model
        tmpdir, calc, config, mliap_path = self._prepare_mliap_model(
            fake_model_training_session, compile, modifiers_str
        )

        # run LAMMPS test
        self._run_lammps_mliap_test(
            tmpdir, calc, config, mliap_path, model_dtype, mliap_tol
        )
