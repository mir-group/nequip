from typing import Union, Optional, Callable, Dict
import warnings
import torch

import ase.data
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper
import nequip.scripts.deploy


def nequip_calculator(model, **kwargs):
    """Build ASE Calculator directly from deployed model."""
    return NequIPCalculator.from_deployed_model(model, **kwargs)


class NequIPCalculator(Calculator):
    """NequIP ASE Calculator.

    .. warning::

        If you are running MD with custom species, please make sure to set the correct masses for ASE.

    """

    implemented_properties = ["energy", "energies", "forces", "stress", "free_energy"]

    def __init__(
        self,
        model: torch.jit.ScriptModule,
        r_max: float,
        device: Union[str, torch.device],
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        transform: Callable = lambda x: x,
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.model = model
        assert isinstance(
            model, torch.nn.Module
        ), "To build a NequIPCalculator from a deployed model, use NequIPCalculator.from_deployed_model"
        self.r_max = r_max
        self.device = device
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.transform = transform

    @classmethod
    def from_deployed_model(
        cls,
        model_path,
        device: Union[str, torch.device] = "cpu",
        species_to_type_name: Optional[Dict[str, str]] = None,
        set_global_options: Union[str, bool] = "warn",
        **kwargs
    ):
        # load model
        model, metadata = nequip.scripts.deploy.load_deployed_model(
            model_path=model_path,
            device=device,
            set_global_options=set_global_options,
        )
        r_max = float(metadata[nequip.scripts.deploy.R_MAX_KEY])

        # build typemapper
        type_names = metadata[nequip.scripts.deploy.TYPE_NAMES_KEY].split(" ")
        if species_to_type_name is None:
            # Default to species names
            warnings.warn(
                "Trying to use chemical symbols as NequIP type names; this may not be correct for your model! To avoid this warning, please provide `species_to_type_name` explicitly."
            )
            species_to_type_name = {s: s for s in ase.data.chemical_symbols}
        type_name_to_index = {n: i for i, n in enumerate(type_names)}
        chemical_symbol_to_type = {
            sym: type_name_to_index[species_to_type_name[sym]]
            for sym in ase.data.chemical_symbols
            if sym in type_name_to_index
        }
        if len(chemical_symbol_to_type) != len(type_names):
            raise ValueError(
                "The default mapping of chemical symbols as type names didn't make sense; please provide an explicit mapping in `species_to_type_name`"
            )
        transform = TypeMapper(chemical_symbol_to_type=chemical_symbol_to_type)

        # build nequip calculator
        if "transform" in kwargs:
            raise TypeError("`transform` not allowed here")
        return cls(
            model=model, r_max=r_max, device=device, transform=transform, **kwargs
        )

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
        for k in AtomicDataDict.ALL_ENERGY_KEYS:
            if k in data:
                del data[k]
        data = self.transform(data)
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)

        # predict + extract data
        out = self.model(data)
        self.results = {}
        # only store results the model actually computed to avoid KeyErrors
        if AtomicDataDict.TOTAL_ENERGY_KEY in out:
            self.results["energy"] = self.energy_units_to_eV * (
                out[AtomicDataDict.TOTAL_ENERGY_KEY]
                .detach()
                .cpu()
                .numpy()
                .reshape(tuple())
            )
            # "force consistant" energy
            self.results["free_energy"] = self.results["energy"]
        if AtomicDataDict.PER_ATOM_ENERGY_KEY in out:
            self.results["energies"] = self.energy_units_to_eV * (
                out[AtomicDataDict.PER_ATOM_ENERGY_KEY]
                .detach()
                .squeeze(-1)
                .cpu()
                .numpy()
            )
        if AtomicDataDict.FORCE_KEY in out:
            # force has units eng / len:
            self.results["forces"] = (
                self.energy_units_to_eV / self.length_units_to_A
            ) * out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
        if AtomicDataDict.STRESS_KEY in out:
            stress = out[AtomicDataDict.STRESS_KEY].detach().cpu().numpy()
            stress = stress.reshape(3, 3) * (
                self.energy_units_to_eV / self.length_units_to_A**3
            )
            # ase wants voigt format
            stress_voigt = full_3x3_to_voigt_6_stress(stress)
            self.results["stress"] = stress_voigt
