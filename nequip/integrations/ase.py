# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Union, Callable, Dict, List
import torch

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from nequip.data import AtomicDataDict, from_ase
from .mixins import _IntegrationLoaderMixin


class NequIPCalculator(_IntegrationLoaderMixin, Calculator):
    """NequIP framework ASE Calculator.

    This ASE Calculator is compatible with models from the NequIP framework, including NequIP and Allegro models.

    The recommended way to use this Calculator is with a compiled model, i.e. ``nequip-compile`` the model and load it into the Calculator with ``NequIPCalculator.from_compiled_model(...)``. If one uses ``--mode aotinductor`` during ``nequip-compile``, it is important to use the flag ``--target ase`` for the compiled model file to work with this ASE Calculator.

    .. warning::

        If you are running MD with custom species, please make sure to set the correct masses for ASE.

    Args:
        model: a model in the NequIP framework
        device (str/torch.device): device for model to evaluate on, e.g. ``cpu`` or ``cuda``
        energy_units_to_eV (float): energy conversion factor (default ``1.0``)
        length_units_to_A (float): length units conversion factor (default ``1.0``)
        transforms (List[Callable]): list of data transforms
    """

    implemented_properties = ["energy", "energies", "forces", "stress", "free_energy"]

    @classmethod
    def _get_aoti_compile_target(cls) -> Dict:
        from nequip.scripts._compile_utils import COMPILE_TARGET_DICT, AOTI_ASE_TARGET

        return COMPILE_TARGET_DICT[AOTI_ASE_TARGET]

    def __init__(
        self,
        model: torch.nn.Module,
        device: Union[str, torch.device],
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        transforms: List[Callable] = [],
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        # === handle model ===
        assert not model.training, (
            "make sure to call .eval() on model before building NequIPCalculator"
        )

        # === handle device ===
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model = model.to(self.device)

        # === data details ===
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.transforms = transforms

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """"""
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        data = self.atoms_to_data(atoms)
        out = self.call_model(data)
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
            # "force consistent" energy
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

        self.save_extra_outputs(out)

    def atoms_to_data(self, atoms: Atoms) -> AtomicDataDict.Type:
        data = from_ase(atoms)
        for t in self.transforms:
            data = t(data)
        return AtomicDataDict.to_(data, self.device)

    def call_model(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        return self.model(data)

    def save_extra_outputs(self, out: AtomicDataDict.Type):
        # subclasses can implement this method to process extra outputs without code duplication
        pass
