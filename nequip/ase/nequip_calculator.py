# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Union, Optional, Callable, Dict, List
import warnings
import torch

from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from nequip.train.lightning import _SOLE_MODEL_KEY
from nequip.nn import graph_model
from nequip.model.saved_models import ModelFromCheckpoint, ModelFromPackage
from nequip.data import AtomicDataDict, from_ase
from nequip.data.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
)
from nequip.utils.global_state import set_global_state


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
    def _handle_chemical_symbols(
        cls,
        chemical_symbols: Optional[Union[List[str], Dict[str, str]]],
        type_names: List[str],
    ) -> Union[List[str], Dict[str, str]]:
        """Handle chemical symbols fallback to type names with warning."""
        if chemical_symbols is None:
            warnings.warn(
                "Trying to use model type names as chemical symbols; this may not be correct for your model (and may cause an error if model type names are not chemical symbols)! To avoid this warning, please provide `chemical_symbols` explicitly."
            )
            chemical_symbols = type_names
        return chemical_symbols

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
    def from_compiled_model(
        cls,
        compile_path: str,
        device: Union[str, torch.device] = "cpu",
        chemical_symbols: Optional[Union[List[str], Dict[str, str]]] = None,
        **kwargs,
    ):
        """Creates a :class:`~nequip.ase.NequIPCalculator` from a compiled model file.

        Args:
            compile_path (str): path to compiled model file
            device (torch.device): the device to use
            chemical_symbols (List[str] or Dict[str, str]): mapping between chemical symbols and model type names
        """
        from nequip.model.inference_models import load_compiled_model
        from nequip.scripts._compile_utils import PAIR_NEQUIP_INPUTS, ASE_OUTPUTS

        model, metadata = load_compiled_model(
            compile_path, device, PAIR_NEQUIP_INPUTS, ASE_OUTPUTS
        )

        # extract r_max and type_names for transforms
        r_max = metadata[graph_model.R_MAX_KEY]
        type_names = metadata[graph_model.TYPE_NAMES_KEY]
        # create neighbor list transform with per-edge-type cutoffs if available
        neighbor_transform = _create_neighbor_transform(metadata, r_max, type_names)

        # use `type_names` metadata as substitute for `chemical_symbols` if latter not provided
        chemical_symbols = cls._handle_chemical_symbols(chemical_symbols, type_names)

        return cls(
            model=model,
            device=device,
            transforms=[
                ChemicalSpeciesToAtomTypeMapper(chemical_symbols),
                neighbor_transform,
            ],
            **kwargs,
        )

    @classmethod
    def _from_checkpoint_model(
        cls,
        ckpt_path: str,
        device: Union[str, torch.device] = "cpu",
        chemical_symbols: Optional[Union[List[str], Dict[str, str]]] = None,
        allow_tf32: bool = False,
        model_name: str = _SOLE_MODEL_KEY,
        **kwargs,
    ):
        """Creates a :class:`~nequip.ase.NequIPCalculator` from a checkpoint file.

        .. note::
            This method is private and intended for internal testing only.
            Users should use `from_compiled_model` instead.

        Args:
            ckpt_path (str): path to checkpoint file
            device (torch.device): the device to use
            chemical_symbols (List[str] or Dict[str, str]): mapping between chemical symbols and model type names
            allow_tf32 (bool): whether to allow TensorFloat32 operations (default ``False``)
        """
        return cls._from_save(
            save_path=ckpt_path,
            model_getter=ModelFromCheckpoint,
            device=device,
            chemical_symbols=chemical_symbols,
            allow_tf32=allow_tf32,
            model_name=model_name,
            **kwargs,
        )

    @classmethod
    def _from_packaged_model(
        cls,
        package_path: str,
        device: Union[str, torch.device] = "cpu",
        chemical_symbols: Optional[Union[List[str], Dict[str, str]]] = None,
        allow_tf32: bool = False,
        model_name: str = _SOLE_MODEL_KEY,
        **kwargs,
    ):
        """Creates a :class:`~nequip.ase.NequIPCalculator` from a package file.

        .. note::
            This method is private and intended for internal testing only.
            Users should use `from_compiled_model` instead.

        Args:
            package_path (str): path to packaged model
            device (torch.device): the device to use
            chemical_symbols (List[str] or Dict[str, str]): mapping between chemical symbols and model type names
            allow_tf32 (bool): whether to allow TensorFloat32 operations (default ``False``)
        """
        return cls._from_save(
            save_path=package_path,
            model_getter=ModelFromPackage,
            device=device,
            chemical_symbols=chemical_symbols,
            allow_tf32=allow_tf32,
            model_name=model_name,
            **kwargs,
        )

    @classmethod
    def _from_save(
        cls,
        save_path: str,
        model_getter: Callable,
        device: Union[str, torch.device] = "cpu",
        chemical_symbols: Optional[Union[List[str], Dict[str, str]]] = None,
        allow_tf32: bool = False,
        model_name: str = _SOLE_MODEL_KEY,
        **kwargs,
    ):
        # === set global state ===
        set_global_state(allow_tf32=allow_tf32)

        # === build model ===
        model: torch.nn.ModuleDict = model_getter(save_path)
        model: graph_model.GraphModel = model[model_name]
        model.eval()
        model.to(device)

        r_max = float(model.metadata[graph_model.R_MAX_KEY])
        type_names = model.metadata[graph_model.TYPE_NAMES_KEY].split(" ")

        # create neighbor list transform with per-edge-type cutoffs if available
        neighbor_transform = _create_neighbor_transform(
            model.metadata, r_max, type_names
        )

        chemical_symbols = cls._handle_chemical_symbols(chemical_symbols, type_names)

        # build nequip calculator
        if "transforms" in kwargs:
            raise KeyError("`transforms` not allowed here")

        return cls(
            model=model,
            device=device,
            transforms=[
                ChemicalSpeciesToAtomTypeMapper(chemical_symbols),
                neighbor_transform,
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


def _create_neighbor_transform(
    metadata: dict, r_max: float, type_names: List[str]
) -> NeighborListTransform:
    """Create NeighborListTransform with per-edge-type cutoffs if available."""
    if metadata.get(graph_model.PER_EDGE_TYPE_CUTOFF_KEY, None) is not None:
        per_edge_type_cutoff = metadata[graph_model.PER_EDGE_TYPE_CUTOFF_KEY]
        if isinstance(per_edge_type_cutoff, str):
            from nequip.nn.embedding.utils import parse_per_edge_type_cutoff_metadata

            per_edge_type_cutoff = parse_per_edge_type_cutoff_metadata(
                per_edge_type_cutoff, type_names
            )

        return NeighborListTransform(
            r_max=r_max,
            per_edge_type_cutoff=per_edge_type_cutoff,
            type_names=type_names,
        )
    else:
        return NeighborListTransform(r_max=r_max)
