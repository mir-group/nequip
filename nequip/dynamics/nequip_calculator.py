import torch
import numpy as np

from ase.calculators.calculator import Calculator, all_changes

from nequip.data import AtomicData, AtomicDataDict

class NequIPCalculator(Calculator):
    """ NequIP ASE Calculator. """
    implemented_properties = ['energy', 'forces']

    def __init__(
            self,
            predictor,
            r_max,
            device,
            force_units_to_eva=1.,
            **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.predictor = predictor
        self.r_max = r_max
        self.device = device
        self.force_units_to_eva = force_units_to_eva

    def calculate(
            self,
            atoms=None,
            properties=["energy"],
            system_changes=all_changes
    ):
        """
        Calculate properties.

        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        data = AtomicData.from_ase(
            atoms=atoms,
            r_max=self.r_max
        )

        data = data.to(self.device)

        # predict + extract data
        out = self.predictor(AtomicData.to_AtomicDataDict(data))
        forces = out['forces'].detach().cpu().numpy()
        energy = out['total_energy'].detach().cpu().item()

        # store results
        self.results = {
            'energy': energy * self.force_units_to_eva,
            'forces': forces * self.force_units_to_eva
        }
