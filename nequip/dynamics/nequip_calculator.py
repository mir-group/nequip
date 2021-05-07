from typing import Union
import torch

from ase.calculators.calculator import Calculator, all_changes

from nequip.data import AtomicData, AtomicDataDict
import nequip.scripts.deploy


class NequIPCalculator(Calculator):
    """NequIP ASE Calculator."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model: torch.jit.ScriptModule,
        r_max: float,
        device: Union[str, torch.device],
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.model = model
        self.r_max = r_max
        self.device = device
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A

    @classmethod
    def from_deployed_model(
        cls, model_path, device: Union[str, torch.device] = "cpu", **kwargs
    ):
        # load model
        model, metadata = nequip.scripts.deploy.load_deployed_model(
            model_path=model_path, device=device
        )
        r_max = float(metadata[nequip.scripts.deploy.R_MAX_KEY])

        # build nequip calculator
        return cls(model=model, r_max=r_max, device=device, **kwargs)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
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
        data = AtomicData.from_ase(atoms=atoms, r_max=self.r_max)

        data = data.to(self.device)

        # predict + extract data
        out = self.model(AtomicData.to_AtomicDataDict(data))
        forces = out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
        energy = out[AtomicDataDict.TOTAL_ENERGY_KEY].detach().cpu().item()

        # store results
        self.results = {
            "energy": energy * self.energy_units_to_eV,
            # force has units eng / len:
            "forces": forces * (self.energy_units_to_eV / self.length_units_to_A),
        }
