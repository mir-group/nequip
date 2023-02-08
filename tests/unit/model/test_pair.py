import pytest

import textwrap
import tempfile
import os
import sys
import subprocess
import numpy as np
from pathlib import Path

import ase.io
import ase.data

import torch

from nequip.data import (
    dataset_from_config,
    AtomicInMemoryDataset,
    AtomicData,
    AtomicDataDict,
)
from nequip.model import model_from_config
from nequip.nn import GraphModel
from nequip.utils import Config


def _check_and_print(retcode):
    __tracebackhide__ = True
    if retcode.returncode:
        if len(retcode.stdout) > 0:
            print(retcode.stdout.decode("ascii"))
        if len(retcode.stderr) > 0:
            print(retcode.stderr.decode("ascii"), file=sys.stderr)
        retcode.check_returncode()


@pytest.mark.skipif(
    "LAMMPS" not in os.environ,
    reason="test_zbl requires a LAMMPS installation pointed to by the LAMMPS environment variable",
)
def test_zbl(float_tolerance, BENCHMARK_ROOT):
    config = textwrap.dedent(
        f"""
        root: results/
        run_name: minimal-pair
        seed: 123

        model_builders:
         - PairPotential
         - StressForceOutput
         - RescaleEnergyEtc

        pair_style: ZBL

        dataset: npz                                                                       # type of data set, can be npz or ase
        dataset_url: http://quantum-machine.org/gdml/data/npz/aspirin_ccsd.zip             # url to download the npz. optional
        dataset_file_name: {BENCHMARK_ROOT}/aspirin_ccsd-train.npz                         # path to data set file
        key_mapping:
          z: atomic_numbers                                                                # atomic species, integers
          E: total_energy                                                                  # total potential eneriges to train to
          F: forces                                                                        # atomic forces to train to
          R: pos                                                                           # raw atomic positions
        npz_fixed_field_keys:                                                              # fields that are repeated across different examples
         - atomic_numbers
        r_max: 4.0
        dataset_statistics_stride: 1

        chemical_symbols:
         - H
         - O
         - C
        """
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(tmpdir + "/config.yaml", "w") as f:
            f.write(config)
        config = Config.from_file(tmpdir + "/config.yaml")
        r_max: float = config["r_max"]

        dataset: AtomicInMemoryDataset = dataset_from_config(config)
        dataset = dataset.index_select(list(range(10)))  # just ten frames
        model: GraphModel = model_from_config(
            config=config, initialize=True, dataset=dataset, deploy=False
        )

        # note that ASE outputs lammps types in alphabetical order of chemical symbols
        # since we use chem symbols in this test, just put the same
        sym_to_lammps_types = dict(
            zip(
                sorted(set(config["chemical_symbols"])),
                range(1, len(config["chemical_symbols"]) + 1),
            )
        )
        pair_coeff = []
        for sym in config["chemical_symbols"]:
            pair_coeff.append(
                f"pair_coeff {sym_to_lammps_types[sym]} {sym_to_lammps_types[sym]} {ase.data.atomic_numbers[sym]:.1f} {ase.data.atomic_numbers[sym]:.1f}"
            )
        pair_coeff = "\n".join(pair_coeff)

        newline = "\n"
        PRECISION_CONST: float = 1e6
        lmp_in = textwrap.dedent(
            f"""
            units		metal
            atom_style	atomic
            thermo 1

            boundary s s s

            read_data structure.data

            pair_style zbl {r_max} {r_max}  # don't use switching function
            {pair_coeff}
{newline.join('        mass  %i 1.0' % i for i in range(1, len(config["chemical_symbols"]) + 1))}

            neighbor	1.0 bin
            neigh_modify    delay 0 every 1 check no
            fix		1 all nve
            timestep	0.001

            compute atomicenergies all pe/atom
            compute totalatomicenergy all reduce sum c_atomicenergies
            compute stress all pressure NULL virial  # NULL means without temperature contribution

            thermo_style custom step time temp pe c_totalatomicenergy etotal press spcpu cpuremain c_stress[*]
            run 0
            print "$({PRECISION_CONST} * c_stress[1]) $({PRECISION_CONST} * c_stress[2]) $({PRECISION_CONST} * c_stress[3]) $({PRECISION_CONST} * c_stress[4]) $({PRECISION_CONST} * c_stress[5]) $({PRECISION_CONST} * c_stress[6])" file stress.dat
            print $({PRECISION_CONST} * pe) file pe.dat
            print $({PRECISION_CONST} * c_totalatomicenergy) file totalatomicenergy.dat
            write_dump all custom output.dump id type x y z fx fy fz c_atomicenergies modify format float %20.15g
            """
        )

        # save out the LAMMPS input:
        infile_path = tmpdir + "/test_repro.in"
        with open(infile_path, "w") as f:
            f.write(lmp_in)

        for structure in [dataset[i] for i in range(10)]:
            struc_ase = structure.to_ase(type_mapper=dataset.type_mapper)
            struc_ase.cell = np.eye(3) * 100
            struc_ase.positions += 50
            ase.io.write(
                tmpdir + "/structure.data",
                struc_ase,
                format="lammps-data",
            )

            retcode = subprocess.run(
                [os.environ["LAMMPS"], "-in", infile_path],
                cwd=tmpdir,
                env=os.environ,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _check_and_print(retcode)

            # load dumped data
            lammps_result = ase.io.read(
                tmpdir + "/output.dump", format="lammps-dump-text"
            )

            # --- now check the OUTPUTS ---
            nequip_out = model(AtomicData.to_AtomicDataDict(structure))
            with torch.no_grad():
                assert np.allclose(
                    nequip_out[AtomicDataDict.FORCE_KEY],
                    lammps_result.get_forces(),
                    atol=1e-4,
                )
                assert np.allclose(
                    nequip_out[AtomicDataDict.PER_ATOM_ENERGY_KEY],
                    lammps_result.arrays["c_atomicenergies"].reshape(-1),
                    atol=5e-5,
                )

                # check system quantities
                lammps_pe = (
                    float(Path(tmpdir + "/pe.dat").read_text()) / PRECISION_CONST
                )
                lammps_totalatomicenergy = (
                    float(Path(tmpdir + "/totalatomicenergy.dat").read_text())
                    / PRECISION_CONST
                )
                assert np.allclose(lammps_pe, lammps_totalatomicenergy)
                assert np.allclose(
                    nequip_out[AtomicDataDict.TOTAL_ENERGY_KEY],
                    lammps_pe,
                    atol=1e-6,
                )
