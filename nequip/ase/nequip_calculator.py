from typing import Union, Optional, Callable, Dict, List
import warnings
import torch

from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from nequip.nn import graph_model
from nequip.model.from_save import ModelFromCheckpoint, ModelFromPackage
from nequip.data import AtomicDataDict, from_ase
from nequip.data.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
)


class NequIPCalculator(Calculator):
    """NequIP ASE Calculator.

    .. warning::

        If you are running MD with custom species, please make sure to set the correct masses for ASE.

    Args:
        model (nequip.nn.GraphModel): NequIP GraphModel object
        device (str/torch.device): device for model to evlauate on, e.g. ``cpu`` or ``cuda``
        transforms (List[Callable]): list of data transforms
    """

    implemented_properties = ["energy", "energies", "forces", "stress", "free_energy"]

    def __init__(
        self,
        model: graph_model.GraphModel,
        device: Union[str, torch.device],
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        transforms: List[Callable] = [],
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        # === handle model ===
        assert (
            not model.training
        ), "make sure to call .eval() on model before building NequIPCalculator"
        self.model = model

        # === handle device ===
        self.device = device

        # === data details ===
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.transforms = transforms

    @classmethod
    def from_checkpoint_model(
        cls,
        ckpt_path: str,
        device: Union[str, torch.device] = "cpu",
        chemical_symbols: Optional[Union[List[str], Dict[str, str]]] = None,
        set_global_options: Union[str, bool] = "warn",
        **kwargs,
    ):
        """Creates a NequIPCalculator from a checkpoint file.

        Args:
            ckpt_path (str): path to checkpoint file
            device (torch.device): the device to use
            chemical_symbols (List[str] or Dict[str, str]): mapping between chemical symbols and model type names
            set_global_options (str/bool): ``True``, ``False``, or ``"warn"``
        """
        return cls._from_save(
            save_path=ckpt_path,
            model_getter=ModelFromCheckpoint,
            device=device,
            chemical_symbols=chemical_symbols,
            set_global_options=set_global_options,
            **kwargs,
        )

    @classmethod
    def from_packaged_model(
        cls,
        package_path: str,
        device: Union[str, torch.device] = "cpu",
        chemical_symbols: Optional[Union[List[str], Dict[str, str]]] = None,
        set_global_options: Union[str, bool] = "warn",
        **kwargs,
    ):
        """Creates a NequIPCalculator from a package file.

        Args:
            package_path (str): path to packaged model
            device (torch.device): the device to use
            chemical_symbols (List[str] or Dict[str, str]): mapping between chemical symbols and model type names
            set_global_options (str/bool): ``True``, ``False``, or ``"warn"``
        """
        return cls._from_save(
            save_path=package_path,
            model_getter=ModelFromPackage,
            device=device,
            chemical_symbols=chemical_symbols,
            set_global_options=set_global_options,
            **kwargs,
        )

    @classmethod
    def _from_save(
        cls,
        save_path: str,
        model_getter: Callable,
        device: Union[str, torch.device] = "cpu",
        chemical_symbols: Optional[Union[List[str], Dict[str, str]]] = None,
        set_global_options: Union[str, bool] = "warn",
        **kwargs,
    ):
        model = model_getter(save_path, set_global_options)
        model.eval()
        model.to(device)

        r_max = float(model.metadata[graph_model.R_MAX_KEY])

        if chemical_symbols is None:
            type_names = model.metadata[graph_model.TYPE_NAMES_KEY].split(" ")
            # Default to species names
            warnings.warn(
                "Trying to use model type names as chemical symbols; this may not be correct for your model (and may cause an error if model type names are not chemical symbols)! To avoid this warning, please provide `chemical_symbols` explicitly."
            )
            chemical_symbols = type_names

        # build nequip calculator
        if "transforms" in kwargs:
            raise KeyError("`transforms` not allowed here")

        return cls(
            model=model,
            device=device,
            transforms=[
                ChemicalSpeciesToAtomTypeMapper(chemical_symbols),
                NeighborListTransform(r_max=r_max),
            ],
            **kwargs,
        )

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """"""
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # === prepare data ===
        data = from_ase(atoms)
        for t in self.transforms:
            data = t(data)
        data = AtomicDataDict.to_(data, self.device)

        # === predict + extract data ===
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
