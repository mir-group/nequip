import os
import logging

from ase.io import write


def save_to_xyz(atoms, logdir, prefix=''):
    """
    Save structure to extended xyz file.

    :param atoms: ase.Atoms object to save
    :param logdir, str, path/to/logging/directory
    :param prefix: str, prefix to use for storing xyz files
    """
    write(
        filename=os.path.join(
            os.path.join(
                logdir,
                'xyz_strucs/'
            ),
            prefix + '.xyz'
        ),
        images=atoms,
        format='extxyz',
        append=True
    )


def write_ase_md_config(curr_atoms, curr_step, dt):
    """ Write time, positions, forces, and atomic kinetic energies to log file.

    :param curr_atoms: ase.Atoms object, current system to log
    :param curr_step: int, current step / frame in MD simulation
    :param dt: float, MD time step
    """
    parsed_temperature = curr_atoms.get_temperature()

    # frame
    log_txt = "-------------------- \n-Frame: {}".format(str(curr_step))
    log_txt += " Simulation Time: {:.6f}\t Temperature: {:.8f} K\n\n".format(
        dt * curr_step,
        parsed_temperature
    )

    # header
    log_txt += 'El \t\t\t\t'
    log_txt += 'Position [A] \t\t\t\t\t\t\t\t   '
    log_txt += 'Predicted Force [eV/A]\n'

    forces = curr_atoms.get_forces()
    atomic_numbers = curr_atoms.get_atomic_numbers()
    positions = curr_atoms.get_positions()

    # write atom by atom
    for i in range(len(curr_atoms)):
        log_txt += '{}\t '.format(str(atomic_numbers[i]))

        for j in range(3):
            log_txt += '{:.8f}  \t'.format(positions[i][j])

        log_txt += '\t\t'

        for j in range(3):
            log_txt += '{:.8f}  \t'.format(forces[i][j])
        log_txt += '\n'

    logging.info(log_txt)