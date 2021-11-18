import logging
import os
import time
import argparse
import numpy as np

import torch

from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.velocitydistribution import Stationary, ZeroRotation

from nequip.ase import NequIPCalculator
from nequip.ase import NoseHoover


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


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Run Nose-Hoover MD using a deployed NequIP model."
    )
    parser.add_argument("model", help="The deployed NequIP model.", type=str)
    parser.add_argument(
        "initial_xyz", help="Initial positions in XYZ format.", type=str
    )
    parser.add_argument("logdir", help="Output directory.", type=str)
    parser.add_argument("--seed", help="Seed for PRNGs.", type=int, default=0)
    parser.add_argument(
        "--log-frequency", help="Log every n steps.", type=int, default=1000
    )
    parser.add_argument(
        "--save-frequency", help="Save every n steps.", type=int, default=1000
    )
    parser.add_argument(
        "--energy-units-to-eV",
        help="Conversion factor from model energy units into eV",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--length-units-to-A",
        help="Conversion factor from model length units into Angstrom",
        type=float,
        default=1.0,
    )
    parser.add_argument("--temperature", help="Temperature", type=float, default=300.0)
    parser.add_argument("--dt", help="Timestep (fs)", type=float, default=1.0)
    parser.add_argument(
        "--n-steps", help="Number of steps to run", type=int, default=500000
    )
    parser.add_argument("--nvt-q", type=float, default=43.06225052549201)
    args = parser.parse_args(args=args)

    logfilename = os.path.join(args.logdir, f"ase_md_run_{time.time()}.log")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
        os.makedirs(os.path.join(args.logdir, "xyz_strucs"))

    logging.basicConfig(filename=logfilename, format="%(message)s", level=logging.INFO)

    # load atoms
    atoms = read(args.initial_xyz, index=0)

    # build nequip calculator
    calc = NequIPCalculator.from_deployed_model(
        model_path=args.model,
        device=device,
        energy_units_to_eV=args.energy_units_to_eV,
        length_units_to_A=args.length_units_to_A,
    )

    atoms.set_calculator(calc=calc)

    # set starting temperature
    MaxwellBoltzmannDistribution(atoms=atoms, temp=args.temperature * units.kB)

    ZeroRotation(atoms)
    Stationary(atoms)

    nvt_dyn = NoseHoover(
        atoms=atoms,
        timestep=args.dt * units.fs,
        temperature=args.temperature,
        nvt_q=args.nvt_q,
    )

    # log first frame
    logging.info(
        f"\n\nStarting dynamics with Nose-Hoover Thermostat with nvt_q: {args.nvt_q}\n\n"
    )
    write_ase_md_config(curr_atoms=atoms, curr_step=0, dt=args.dt)
    logging.info(f"COM [A]: {atoms.get_center_of_mass()}\n")

    save_to_xyz(atoms, logdir=args.logdir, prefix="nvt_")

    for i in range(1, args.n_steps):
        nvt_dyn.run(steps=1)

        if not i % args.log_frequency:
            write_ase_md_config(curr_atoms=atoms, curr_step=i, dt=args.dt)

            logging.info(f"COM [A]: {atoms.get_center_of_mass()}\n")

        # append current structure to xyz file
        if not i % args.save_frequency:
            save_to_xyz(atoms, logdir=args.logdir, prefix="nvt_")

    print("finished...")


if __name__ == "__main__":
    main()
