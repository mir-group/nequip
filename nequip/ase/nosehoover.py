"""Custom Nose-Hoover NVT thermostat based on ASE.

This code was originally written by Jonathan Mailoa based on these notes:

    https://www2.ph.ed.ac.uk/~dmarendu/MVP/MVP03.pdf

It was then adapted by Simon Batzner to be used within ASE. Parts of the overall outline of the class are also based on the Langevin class in ASE.
"""

import numpy as np

from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import Stationary, ZeroRotation
from ase import units


class NoseHoover(MolecularDynamics):
    """Nose-Hoover (constant N, V, T) molecular dynamics.

    Usage: NoseHoover(atoms, dt, temperature)

    atoms
        The list of atoms.

    timestep
        The time step.

    temperature
        Target temperature of the MD run in [K]

    nvt_q
        Q in the Nose-Hoover equations

    Example Usage:

        nvt_dyn = NoseHoover(
            atoms=atoms,
            timestep=0.5 * units.fs,
            temperature=300.,
            nvt_q=334.
        )

    """

    def __init__(
        self,
        atoms,
        timestep,
        temperature,
        nvt_q,
        trajectory=None,
        logfile=None,
        loginterval=1,
        append_trajectory=False,
    ):
        # set angular and com momentum to zero, necessary for nose-hoover dynamics.
        ZeroRotation(atoms)
        Stationary(atoms)

        # thermostat parameters
        self.temp = temperature
        self.nvt_q = nvt_q
        self.dt = timestep
        self.dtdt = np.power(self.dt, 2)
        self.nvt_bath = 0.0

        self.natoms = len(atoms)

        MolecularDynamics.__init__(
            self,
            atoms,
            timestep,
            trajectory,
            logfile,
            loginterval,
            append_trajectory=append_trajectory,
        )

    def step(self):
        """Perform a MD step."""
        masses = self.atoms.get_masses()

        modified_acc = (
            self.atoms.get_forces() / masses[:, np.newaxis]
            - self.nvt_bath * self.atoms.get_velocities()
        )
        pos_fullstep = (
            self.atoms.get_positions()
            + self.dt * self.atoms.get_velocities()
            + 0.5 * self.dtdt * modified_acc
        )
        vel_halfstep = self.atoms.get_velocities() + 0.5 * self.dt * modified_acc

        self.atoms.set_positions(pos_fullstep)

        e_kin_diff = 0.5 * (
            np.sum(masses * np.sum(self.atoms.get_velocities() ** 2, axis=1))
            - (3 * self.natoms + 1) * units.kB * self.temp
        )

        nvt_bath_halfstep = self.nvt_bath + 0.5 * self.dt * e_kin_diff / self.nvt_q
        e_kin_diff_halfstep = 0.5 * (
            np.sum(masses * np.sum(vel_halfstep**2, axis=1))
            - (3 * self.natoms + 1) * units.kB * self.temp
        )
        self.nvt_bath = (
            nvt_bath_halfstep + 0.5 * self.dt * e_kin_diff_halfstep / self.nvt_q
        )
        self.atoms.set_velocities(
            (
                vel_halfstep
                + 0.5 * self.dt * (self.atoms.get_forces() / masses[:, np.newaxis])
            )
            / (1 + 0.5 * self.dt * self.nvt_bath)
        )
