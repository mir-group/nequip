import os
import subprocess
import textwrap
from pathlib import Path
import tempfile
import warnings
import sys

import ase.io
import numpy as np
import torch
import pytest

from nequip.data import to_ase
from nequip.utils.global_dtype import _GLOBAL_DTYPE
from nequip.ase import NequIPCalculator
from nequip.utils.versions import _TORCH_GE_2_6
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate

# MLIAP test configuration
TESTS_DIR = Path(__file__).resolve().parent
LAMMPS = os.environ.get("LAMMPS", "lmp")
LAMMPS_ENV_PREFIX = os.environ.get("LAMMPS_ENV_PREFIX", "")

# check if LAMMPS is available
HAS_LAMMPS = False
HAS_KOKKOS = False
HAS_KOKKOS_CUDA = False
HAS_MLIAP = False

# Check for OpenEquivariance availability
HAS_OPENEQUIVARIANCE = False
try:
    import openequivariance  # noqa: F401

    HAS_OPENEQUIVARIANCE = True
except ImportError:
    pass

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
    warnings.warn(
        f"LAMMPS binary '{LAMMPS}' not found or not executable. "
        "Set LAMMPS environment variable to specify path. "
        "Skipping MLIAP integration tests."
    )

if HAS_LAMMPS:
    if not HAS_KOKKOS:
        warnings.warn(
            "Not testing nequip MLIAP with Kokkos since it wasn't built with it"
        )
    if HAS_KOKKOS and torch.cuda.is_available() and not HAS_KOKKOS_CUDA:
        warnings.warn("Kokkos not built with CUDA even though CUDA is available")
    if not HAS_MLIAP:
        warnings.warn("Not testing nequip MLIAP since LAMMPS wasn't built with ML-IAP")


@pytest.fixture(
    params=[
        ("Cu-cubic.xyz", ["Cu"], 5.0),
        ("aspirin.xyz", ["C", "H", "O"], 5.0),
    ],
    scope="session",
)
def dataset_options(request):
    out = dict(
        zip(
            ["dataset_file_name", "chemical_symbols", "cutoff_radius"],
            request.param,
        )
    )
    out["dataset_file_name"] = TESTS_DIR / ("test_data/" + out["dataset_file_name"])
    return out


@pytest.fixture(
    params=["float32", "float64"],
    scope="session",
)
def model_dtype(request):
    return request.param


@pytest.fixture(
    params=[None]
    + (
        []
        if not (HAS_OPENEQUIVARIANCE and _TORCH_GE_2_6 and torch.cuda.is_available())
        else ["openequivariance"]
    ),
    scope="session",
)
def modifiers(request):
    if request.param is None:
        return ""
    elif request.param == "openequivariance":
        return "--modifiers enable_OpenEquivariance"
    else:
        raise ValueError(f"Unknown modifier parameter: {request.param}")


def _check_and_print(retcode, encoding="ascii"):
    __tracebackhide__ = True
    if retcode.returncode:
        if len(retcode.stdout) > 0:
            print(retcode.stdout.decode(encoding, errors="replace"))
        if len(retcode.stderr) > 0:
            print(retcode.stderr.decode(encoding, errors="replace"), file=sys.stderr)
        retcode.check_returncode()


@pytest.fixture(scope="session")
def trained_nequip_model(model_dtype, dataset_options):
    """Train a NequIP model once and provide paths to checkpoint and package."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield train_nequip_model(tmpdir, model_dtype, dataset_options)


def train_nequip_model(tmpdir, dtype, dataset_options):
    # === setup config from template ===
    config = OmegaConf.load(str(TESTS_DIR / "test_data/nequip_config.yaml"))
    with open_dict(config):
        # the checkpoint file `last.ckpt` will be located in the hydra runtime directory
        # so we set it to the tmpdir
        config["hydra"] = {"run": {"dir": tmpdir}}
        config["model_dtype"] = dtype
        config.update(dataset_options)
    config = OmegaConf.create(config)
    configpath = tmpdir + "/config.yaml"
    OmegaConf.save(config=config, f=configpath)

    # === run `nequip-train` to get checkpoint ===
    retcode = subprocess.run(
        ["nequip-train", "-cn", "config"],
        cwd=tmpdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _check_and_print(retcode)
    checkpoint_path = tmpdir + "/last.ckpt"

    # === create packaged model ===
    package_path = tmpdir + "/packaged_model.nequip.zip"
    retcode = subprocess.run(
        [
            "nequip-package",
            "build",
            checkpoint_path,
            package_path,
        ],
        cwd=tmpdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _check_and_print(retcode)
    assert Path(package_path).is_file(), "`nequip-package` didn't create file"

    # always load calculator from checkpoint for comparison
    calc = NequIPCalculator._from_checkpoint_model(
        checkpoint_path,
        chemical_symbols=config["chemical_symbols"],
    )

    # === get the `test` dataset (5 frames) ===
    torch.set_default_dtype(_GLOBAL_DTYPE)
    datamodule = instantiate(config.data, _recursive_=False)
    datamodule.prepare_data()
    datamodule.setup("test")
    dloader = datamodule.test_dataloader()[0]

    structures = []
    for data in dloader:
        # `to_ase` returns a List because data from datamodule's dataloader is batched (trivially with batch size 1)
        structures += to_ase(data)

    # give them cells even if nonperiodic
    if not all(structures[0].pbc):
        L = 50.0
        for struct in structures:
            struct.cell = L * np.eye(3)
            struct.center()
    for s in structures:
        # wrapping is extremely important for the nequip tests
        s.wrap()
    structures = structures[:1]

    return (
        tmpdir,
        checkpoint_path,
        package_path,
        calc,
        structures,
        config,
    )


def prepare_mliap_model(trained_nequip_model, model_source, compile, modifiers):
    """Prepare MLIAP model from trained model with specified options."""
    (
        tmpdir,
        checkpoint_path,
        package_path,
        calc,
        structures,
        config,
    ) = trained_nequip_model

    # select model path based on source
    model_path = checkpoint_path if model_source == "checkpoint" else package_path

    # prepare compile flag
    compile_flag = "" if compile else "--no-compile"

    # prepare MLIAP file
    mliap_path = (
        tmpdir
        + f"/model_{model_source}_{compile_flag.replace('--', '').replace('-', '_') or 'compiled'}.nequip.lmp.pt"
    )
    command = f"nequip-prepare-lmp-mliap {model_path} {mliap_path} {modifiers} {compile_flag}".strip()
    print(
        f"Preparing MLIAP ({model_source}, {'compiled' if compile else 'no-compile'}): {command}"
    )

    retcode = subprocess.run(
        command,
        shell=True,
        cwd=tmpdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(os.environ),
    )
    _check_and_print(retcode, "utf-8")

    return tmpdir, calc, structures, config, mliap_path


@pytest.mark.skipif(not HAS_LAMMPS, reason="LAMMPS not available")
@pytest.mark.skipif(not HAS_MLIAP, reason="LAMMPS not built with ML-IAP")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_KOKKOS_CUDA, reason="LAMMPS not built with Kokkos CUDA")
@pytest.mark.skipif(not _TORCH_GE_2_6, reason="MLIAP requires torch >= 2.6")
@pytest.mark.parametrize("model_source", ["checkpoint", "package"])
@pytest.mark.parametrize("compile", [True, False])
def test_mliap_gpu_output(
    trained_nequip_model, modifiers, model_dtype, model_source, compile
):
    """Test LAMMPS MLIAP integration on GPU matches direct NequIP calculations."""

    if model_dtype == "float64":
        pytest.skip("Skipping f64 ML-IAP tests.")

    tol = {"float32": 5e-4, "float64": 1e-8}[model_dtype]

    # prepare MLIAP model
    (
        model_tmpdir,
        calc,
        structures,
        config,
        mliap_path,
    ) = prepare_mliap_model(trained_nequip_model, model_source, compile, modifiers)

    structure: ase.Atoms

    num_types = len(config["chemical_symbols"])

    newline = "\n"
    periodic = all(structures[0].pbc)
    PRECISION_CONST: float = 1e6
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
        pair_coeff * * {' '.join(sorted(set(config["chemical_symbols"])))}
        {newline.join([f"mass {i+1} 1.0" for i in range(num_types)])}

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

    for i, structure in enumerate(structures):
        # === run LAMMPS ===
        lammps_tmpdir = model_tmpdir + f"/lammps_run_{i}"
        os.makedirs(lammps_tmpdir, exist_ok=True)

        # write structure using ASE's LAMMPS data format
        ase.io.write(
            lammps_tmpdir + "/structure.data",
            structure,
            format="lammps-data",
        )

        with open(lammps_tmpdir + "/in.lammps", "w") as f:
            f.write(lmp_in)

        # environment variables (following pair_nequip_allegro pattern)
        env = dict(os.environ)
        env["_NEQUIP_LOG_LEVEL"] = "DEBUG"
        env["CUDA_VISIBLE_DEVICES"] = "0"

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
        # uncomment to view LAMMPS output
        print(retcode.stdout.decode("ascii"))
        _check_and_print(retcode)

        # === extract results ===
        # load dumped data
        lammps_result = ase.io.read(
            lammps_tmpdir + "/output.dump", format="lammps-dump-text"
        )
        structure.calc = calc

        # === atomic energies ===
        nequip_atomic_energies = structure.get_potential_energies()
        lammps_atomic_energies = lammps_result.arrays["c_atomicenergies"].reshape(-1)
        np.testing.assert_allclose(
            nequip_atomic_energies,
            lammps_atomic_energies,
            atol=tol,
            rtol=tol,
        )

        # === total energy ===
        lammps_pe = float(Path(lammps_tmpdir + "/pe.dat").read_text()) / PRECISION_CONST
        nequip_pe = structure.get_potential_energy()
        np.testing.assert_allclose(
            lammps_pe,
            nequip_pe,
            atol=tol,
            rtol=tol,
        )

        # === forces ===
        max_force_err = np.max(
            np.abs(structure.get_forces() - lammps_result.get_forces())
        )
        max_force_comp = np.max(np.abs(structure.get_forces()))
        force_rms = np.sqrt(np.mean(np.square(structure.get_forces())))
        np.testing.assert_allclose(
            structure.get_forces(),
            lammps_result.get_forces(),
            atol=tol,
            rtol=tol,
            err_msg=f"Force comparison failed. Max force error: {max_force_err}, Max force component: {max_force_comp}, Force RMS: {force_rms}",
        )
