import logging
import os
import time
import numpy as np
import torch
import nequip

from ase import units
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.velocitydistribution import Stationary, ZeroRotation

from nequip.dynamics.md_utils import save_to_xyz, write_ase_md_config
from nequip.dynamics.nequip_calculator import NequIPCalculator
from nequip.scripts.deploy import load_deployed_model

if __name__ == "__main__":
    np.random.seed(0)
    log_freq = 1
    save_freq = 1
    logdir = './md_runs/aspirin-test/'
    logfilename = os.path.join(logdir, f'ase_md_run_{time.time()}.log')
    prefix = "nvt_langevin"
    filename = '/Users/simonbatzner1/Desktop/Research/Research/Research_Code/nequip/results/aspirin/example-run/deployed.pth'
    atoms_path = '/Users/simonbatzner1/Desktop/Research/Research/databases/md17/aspirin_dft.xyz'
    force_units_to_eva = (units.kcal/units.mol)
    temperature = 300
    dt = 0.5
    friction = 0.01
    langevin_fix_com = True
    n_steps = 500000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(logdir):
        os.makedirs(logdir)
        os.makedirs(os.path.join(logdir, 'xyz_strucs'))

    logging.basicConfig(
        filename=logfilename,
        format="%(message)s",
        level=logging.INFO
    )

    # load model
    model, metadata = load_deployed_model(model_path=filename)

    # TODO: get this from metadata instead
    # float(metadata[nequip.scripts.deploy.R_MAX_KEY])
    r_max = 4.

    # load atoms
    atoms = read(atoms_path, index=0)

    # build nequip calculator
    calc = NequIPCalculator(
        predictor=model,
        r_max=r_max,
        force_units_to_eva=force_units_to_eva,
        device=device
    )

    atoms.set_calculator(calc=calc)

    # run MD
    # set starting temperature
    MaxwellBoltzmannDistribution(
        atoms=atoms,
        temp=temperature * units.kB
    )

    ZeroRotation(atoms)         # zero angular momentum
    Stationary(atoms)           # zero linear momentum

    nvt_dyn = Langevin(
        atoms=atoms,
        timestep=dt * units.fs,
        temperature=temperature * units.kB,
        friction=friction,
        fixcm=langevin_fix_com
    )

    # log first frame
    write_ase_md_config(curr_atoms=atoms, curr_step=0, dt=dt)
    logging.info(f"COM [A]: {atoms.get_center_of_mass()}\n")
    save_to_xyz(atoms, logdir=logdir, prefix=prefix)

    for i in range(1, n_steps):
        nvt_dyn.run(steps=1)

        if not i % log_freq:
            write_ase_md_config(
                curr_atoms=atoms,
                curr_step=i,
                dt=dt
            )

            logging.info(f"COM [A]: {atoms.get_center_of_mass()}\n")

        # append current structure to xyz file
        if not i % save_freq:
            save_to_xyz(
                atoms,
                logdir=logdir,
                prefix=prefix
            )

    print('finished...')