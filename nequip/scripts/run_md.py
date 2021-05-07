import logging
import sys
import os
import time
import numpy as np

import torch

from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.velocitydistribution import Stationary, ZeroRotation

import nequip
from nequip.dynamics.nequip_calculator import NequIPCalculator
from nequip.scripts.deploy import load_deployed_model
from nequip.dynamics.nosehoover import NoseHoover


def save_to_xyz(atoms, logdir, prefix=""):
    """
    Save structure to extended xyz file.

    :param atoms: ase.Atoms object to save
    :param logdir, str, path/to/logging/directory
    :param prefix: str, prefix to use for storing xyz files
    """
    write(
        filename=os.path.join(os.path.join(logdir, "xyz_strucs/"), prefix + ".xyz"),
        images=atoms,
        format="extxyz",
        append=True,
    )


def write_ase_md_config(curr_atoms, curr_step, dt):
    """Write time, positions, forces, and atomic kinetic energies to log file.

    :param curr_atoms: ase.Atoms object, current system to log
    :param curr_step: int, current step / frame in MD simulation
    :param dt: float, MD time step
    """
    parsed_temperature = curr_atoms.get_temperature()

    # frame
    log_txt = "-------------------- \n-Frame: {}".format(str(curr_step))
    log_txt += " Simulation Time: {:.6f}\t Temperature: {:.8f} K\n\n".format(
        dt * curr_step, parsed_temperature
    )

    # header
    log_txt += "El \t\t\t\t"
    log_txt += "Position [A] \t\t\t\t\t\t\t\t   "
    log_txt += "Predicted Force [eV/A]\n"

    forces = curr_atoms.get_forces()
    atomic_numbers = curr_atoms.get_atomic_numbers()
    positions = curr_atoms.get_positions()

    # write atom by atom
    for i in range(len(curr_atoms)):
        log_txt += "{}\t ".format(str(atomic_numbers[i]))

        for j in range(3):
            log_txt += "{:.8f}  \t".format(positions[i][j])

        log_txt += "\t\t"

        for j in range(3):
            log_txt += "{:.8f}  \t".format(forces[i][j])
        log_txt += "\n"

    logging.info(log_txt)


if __name__ == "__main__":
    seed = int(sys.argv[1])

    log_freq = 1000
    save_freq = 1000

    logdir = "./md_runs/lips_example/"
    logfilename = os.path.join(logdir, f"ase_md_run_{time.time()}.log")

    prefix = "nvt_nose_hoover"
    filename = "path/to/deployed/model.pth"
    atoms_path = "path/to/atoms.xyz"
    force_units_to_eV_A = 1.0
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
        force_units_to_eV_A=force_units_to_eV_A,
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
