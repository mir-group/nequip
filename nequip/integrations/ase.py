# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Union, Optional, Callable, Dict, List
import torch

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from nequip.train.lightning import _SOLE_MODEL_KEY
from nequip.nn import graph_model
from nequip.data import AtomicDataDict, from_ase
from nequip.utils.global_state import set_global_state
from .utils import handle_chemical_species_map, basic_transforms


class NequIPCalculator(Calculator):
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
    def get_aoti_compile_target(cls) -> Dict:
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

    @classmethod
    def from_compiled_model(
        cls,
        compile_path: str,
        device: Union[str, torch.device] = "cpu",
        chemical_species_to_atom_type_map: Optional[Union[Dict[str, str], bool]] = None,
        chemical_symbols: Optional[Union[List[str], Dict[str, str]]] = None,
        neighborlist_backend: str = "matscipy",
        **kwargs,
    ):
        """Creates a :class:`~nequip.integrations.ase.NequIPCalculator` from a compiled model file.

        Args:
            compile_path (str): path to compiled model file.
            device (str or torch.device): the device to use (e.g., ``"cpu"`` or ``"cuda"``).
            chemical_species_to_atom_type_map (Dict[str, str] or bool or None): mapping from chemical species to model type names.
                If ``None`` (default), uses identity mapping with warning.
                If ``True``, uses identity mapping without warning.
                If dict, uses the provided mapping.
            neighborlist_backend (str): neighborlist backend to use: ``"ase"``, ``"matscipy"``, or ``"vesin"``
                (default: ``"matscipy"``).
        """
        # TODO: eventually remove this check
        # check for deprecated API usage
        if chemical_symbols is not None:
            raise ValueError(
                "The `chemical_symbols` parameter is no longer supported. "
                "Please use `chemical_species_to_atom_type_map` instead.\n\n"
                "Old usage:\n"
                "  calc = NequIPCalculator.from_compiled_model(\n"
                "      'model.pth',\n"
                "      chemical_symbols=['H', 'C', 'O']\n"
                "  )\n\n"
                "New usage:\n"
                "  calc = NequIPCalculator.from_compiled_model(\n"
                "      'model.pth',\n"
                "      chemical_species_to_atom_type_map={'H': 'H', 'C': 'C', 'O': 'O'}\n"
                "  )\n"
            )

        from nequip.model.inference_models import load_compiled_model

        target = cls.get_aoti_compile_target()
        input_keys = list(target["input"])
        output_keys = list(target["output"])

        model, metadata = load_compiled_model(
            compile_path, device, input_keys, output_keys
        )

        # extract r_max and type_names for transforms
        r_max = metadata[graph_model.R_MAX_KEY]
        type_names = metadata[graph_model.TYPE_NAMES_KEY]

        # use `type_names` metadata as identity map if not provided
        chemical_species_to_atom_type_map = handle_chemical_species_map(
            chemical_species_to_atom_type_map, type_names
        )

        return cls(
            model=model,
            device=device,
            transforms=basic_transforms(
                metadata,
                r_max,
                type_names,
                chemical_species_to_atom_type_map,
                neighborlist_backend=neighborlist_backend,
            ),
            **kwargs,
        )

    @classmethod
    def _from_saved_model(
        cls,
        model_path: str,
        device: Union[str, torch.device] = "cpu",
        chemical_species_to_atom_type_map: Optional[Union[Dict[str, str], bool]] = None,
        allow_tf32: bool = False,
        model_name: str = _SOLE_MODEL_KEY,
        neighborlist_backend: str = "matscipy",
        **kwargs,
    ):
        """Creates a :class:`~nequip.integrations.ase.NequIPCalculator` from a saved model.

        .. note::
            This method is private and intended for internal testing only.
            Users should use `from_compiled_model` instead.

        Args:
            model_path (str): path to a checkpoint file, package file, or nequip.net model ID
                (format: nequip.net:group-name/model-name:version).
            device (str or torch.device): the device to use (e.g., ``"cpu"`` or ``"cuda"``).
            chemical_species_to_atom_type_map (Dict[str, str] or bool or None): mapping from chemical species to model type names.
                If ``None`` (default), uses identity mapping with warning.
                If ``True``, uses identity mapping without warning.
                If dict, uses the provided mapping.
            allow_tf32 (bool): whether to allow TensorFloat32 operations (default ``False``).
            model_name (str): key to select the model from ModuleDict (default for single model case).
            neighborlist_backend (str): neighborlist backend to use: ``"ase"``, ``"matscipy"``, or ``"vesin"``
                (default: ``"matscipy"``).
        """
        from nequip.model.saved_models.load_utils import load_saved_model

        # === set global state ===
        set_global_state(allow_tf32=allow_tf32)

        # === load model using unified loader ===
        model: graph_model.GraphModel = load_saved_model(
            model_path, model_key=model_name
        )
        model.eval()

        r_max = float(model.metadata[graph_model.R_MAX_KEY])
        type_names = model.metadata[graph_model.TYPE_NAMES_KEY].split(" ")

        chemical_species_to_atom_type_map = handle_chemical_species_map(
            chemical_species_to_atom_type_map, type_names
        )

        # build nequip calculator
        if "transforms" in kwargs:
            raise KeyError("`transforms` not allowed here")

        return cls(
            model=model,
            device=device,
            transforms=basic_transforms(
                model.metadata,
                r_max,
                type_names,
                chemical_species_to_atom_type_map,
                neighborlist_backend=neighborlist_backend,
            ),
            **kwargs,
        )

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
