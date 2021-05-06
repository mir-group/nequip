import logging
import sys
import os
import time
import numpy as np
import torch
import nequip

from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.velocitydistribution import Stationary, ZeroRotation

from nequip.dynamics.md_utils import save_to_xyz, write_ase_md_config
from nequip.dynamics.nequip_calculator import NequIPCalculator
from nequip.scripts.deploy import load_deployed_model
from nequip.dynamics.nosehoover import NoseHoover

if __name__ == "__main__":
    seed = int(sys.argv[1])

    log_freq = 1000
    save_freq = 1000

    logdir = "./md_runs/lips_example/"
    logfilename = os.path.join(logdir, f"ase_md_run_{time.time()}.log")

    prefix = "nvt_nose_hoover"
    filename = "path/to/deployed/model.pth"
    atoms_path = "path/to/atoms.xyz"
    force_units_to_eva = 1.0
    temperature = 300
    dt = 1.0
    n_steps = 500000
    nvt_q = 43.06225052549201

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(logdir):
        os.makedirs(logdir)
        os.makedirs(os.path.join(logdir, "xyz_strucs"))

    logging.basicConfig(filename=logfilename, format="%(message)s", level=logging.INFO)

    # load model
    model, metadata = load_deployed_model(model_path=filename, device=device)
    r_max = float(metadata[nequip.scripts.deploy.R_MAX_KEY])

    # load atoms
    atoms = read(atoms_path, index=0)

    # build nequip calculator
    calc = NequIPCalculator(
        predictor=model,
        r_max=r_max,
        device=device,
        force_units_to_eva=force_units_to_eva,
    )

    atoms.set_calculator(calc=calc)

    # set starting temperature
    MaxwellBoltzmannDistribution(atoms=atoms, temp=temperature * units.kB)

    ZeroRotation(atoms)
    Stationary(atoms)

    nvt_dyn = NoseHoover(
        atoms=atoms, timestep=dt * units.fs, temperature=temperature, nvt_q=nvt_q
    )

    # log first frame
    logging.info(
        f"\n\nStarting dynamics with Nose-Hoover Thermostat with nvt_q: {nvt_q}\n\n"
    )
    write_ase_md_config(curr_atoms=atoms, curr_step=0, dt=dt)
    logging.info(f"COM [A]: {atoms.get_center_of_mass()}\n")

    save_to_xyz(atoms, logdir=logdir, prefix="nvt_")

    for i in range(1, n_steps):
        nvt_dyn.run(steps=1)

        if not i % log_freq:
            write_ase_md_config(curr_atoms=atoms, curr_step=i, dt=dt)

            logging.info(f"COM [A]: {atoms.get_center_of_mass()}\n")

        # append current structure to xyz file
        if not i % save_freq:
            save_to_xyz(atoms, logdir=logdir, prefix="nvt_")

    print("finished...")
