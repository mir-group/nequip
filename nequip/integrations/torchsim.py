"""Wrapper for NequIP framework models in torch-sim"""

import warnings
import torch

import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.typing import StateDict

from nequip.data import AtomicDataDict

from nequip.nn import graph_model

from collections.abc import Callable
from pathlib import Path
from typing import Union, List, Optional, Dict


class NequIPTorchSimCalc(ModelInterface):
    """NequIP framework torch-sim calculator.

    This torch-sim calculator is compatible with models from the NequIP framework,
    including NequIP and Allegro models.

    The recommended way to use this calculator is with a compiled model, i.e.
    ``nequip-compile`` the model and load it into the calculator with
    ``NequIPTorchSimCalc.from_compiled_model(...)``.

    Args:
        model (:class:`torch.nn.Module`): a model in the NequIP framework
        r_max (float): cutoff radius for neighbor list construction
        device (str or :class:`torch.device`): device for model to evaluate on,
            e.g. ``"cpu"`` or ``"cuda"`` (default: ``"cpu"``)
        transforms (List[Callable]): list of data transforms
        neighbor_list_fn (Callable): function to compute neighbor lists
            (default: ``vesin_nl_ts``)
        atomic_numbers (:class:`torch.Tensor` or None): atomic numbers with shape
            ``[n_atoms]``. If provided at initialization, cannot be provided
            again during forward pass
        system_idx (:class:`torch.Tensor` or None): batch indices with shape ``[n_atoms]``
            indicating which system each atom belongs to. If not provided
            with ``atomic_numbers``, all atoms are assumed to be in the same system
    """

    def __init__(
        self,
        model: torch.nn.Module,
        r_max: float,
        device: Union[str, torch.device] = "cpu",
        transforms: List[Callable] = [],
        neighbor_list_fn: Callable = vesin_nl_ts,
        atomic_numbers: torch.Tensor | None = None,
        system_idx: torch.Tensor | None = None,
    ) -> None:
        """Initialize the NequIP torch-sim calculator.

        .. note::
            This is a low-level initializer. Users should typically use
            ``from_compiled_model`` instead.
        """
        super().__init__()

        # === set up ModelInterface attributes ===
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        self._dtype = torch.float64  # default dtype for calculations
        self._compute_forces = True
        self._compute_stress = True
        self._memory_scales_with = "n_atoms_x_density"

        self.neighbor_list_fn = neighbor_list_fn

        if not isinstance(model, torch.nn.Module):
            raise TypeError("Invalid model type. Must be a torch.nn.Module.")
        self.model = model.to(self._device)

        self.r_max = torch.tensor(r_max, dtype=self._dtype, device=self._device)

        # move transforms to device (they are torch.nn.Module's)
        self.transforms = [t.to(self._device) for t in transforms]

        # store flag to track if atomic numbers were provided at init
        self.atomic_numbers_in_init = atomic_numbers is not None
        self.n_systems = system_idx.max().item() + 1 if system_idx is not None else 1

        # set up batch information if atomic numbers are provided
        if atomic_numbers is not None:
            if system_idx is None:
                # if batch is not provided, assume all atoms belong to same system
                system_idx = torch.zeros(
                    len(atomic_numbers), dtype=torch.long, device=self._device
                )

            self.setup_from_system_idx(atomic_numbers, system_idx)

    @ModelInterface.compute_forces.setter
    def compute_forces(self, value: bool) -> None:
        """Set whether to compute forces."""
        self._compute_forces = value

    @ModelInterface.compute_stress.setter
    def compute_stress(self, value: bool) -> None:
        """Set whether to compute stress."""
        self._compute_stress = value

    @classmethod
    def _handle_chemical_species_map(
        cls,
        chemical_species_to_atom_type_map: Optional[Union[Dict[str, str], bool]],
        type_names: List[str],
    ) -> Dict[str, str]:
        """Handle chemical species map fallback to identity map with warning.

        Args:
            chemical_species_to_atom_type_map (Dict[str, str] or bool or None): mapping from chemical species to atom type names.
                If ``None`` (default), uses identity mapping with warning.
                If ``True``, uses identity mapping without warning.
                If dict, uses the provided mapping.
            type_names (List[str]): list of model type names.

        Returns:
            Dict[str, str]: the chemical species to atom type mapping.
        """
        if chemical_species_to_atom_type_map is None:
            warnings.warn(
                "Defaulting to using model type names as chemical symbols. "
                "If the model type names correspond exactly to chemical species (e.g., 'H', 'C', 'O'), this is correct. "
                "Otherwise, this is wrong and will cause errors. "
                "To silence this warning, explicitly set `chemical_species_to_atom_type_map=True` for identity mapping, "
                "or provide the correct mapping as a dict."
            )
            chemical_species_to_atom_type_map = {t: t for t in type_names}
        elif chemical_species_to_atom_type_map is True:
            # explicitly requested identity mapping without warning
            chemical_species_to_atom_type_map = {t: t for t in type_names}
        return chemical_species_to_atom_type_map

    @classmethod
    def from_compiled_model(
        cls,
        compile_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        chemical_species_to_atom_type_map: Optional[Union[Dict[str, str], bool]] = None,
        **kwargs,
    ):
        """Creates a :class:`~nequip.integrations.torchsim.NequIPTorchSimCalc` from a compiled model file.

        Args:
            compile_path (str or :class:`pathlib.Path`): path to compiled model file.
            device (str or :class:`torch.device`): the device to use (e.g., ``"cpu"`` or ``"cuda"``).
            chemical_species_to_atom_type_map (Dict[str, str] or bool or None): mapping from chemical species to model type names.
                If ``None`` (default), uses identity mapping with warning.
                If ``True``, uses identity mapping without warning.
                If dict, uses the provided mapping.
            **kwargs: additional arguments passed to :class:`~nequip.integrations.torchsim.NequIPTorchSimCalc`.
        """
        from nequip.model.inference_models import load_compiled_model
        from nequip.scripts._compile_utils import ASE_OUTPUTS, BATCH_INPUTS

        model, metadata = load_compiled_model(
            str(compile_path), device, BATCH_INPUTS, ASE_OUTPUTS
        )

        # extract r_max and type_names from metadata
        r_max = metadata[graph_model.R_MAX_KEY]
        type_names = metadata[graph_model.TYPE_NAMES_KEY]

        # handle chemical species mapping
        chemical_species_to_atom_type_map = cls._handle_chemical_species_map(
            chemical_species_to_atom_type_map, type_names
        )

        # check for transforms in kwargs
        if "transforms" in kwargs:
            raise KeyError("`transforms` not allowed here")

        return cls(
            model=model,
            r_max=r_max,
            device=device,
            transforms=_basic_transforms(
                metadata, r_max, type_names, chemical_species_to_atom_type_map
            ),
            **kwargs,
        )

    @classmethod
    def _from_saved_model(
        cls,
        model_path: str,
        device: Union[str, torch.device] = "cpu",
        chemical_species_to_atom_type_map: Optional[Union[Dict[str, str], bool]] = None,
        **kwargs,
    ):
        """Creates a :class:`~nequip.integrations.torchsim.NequIPTorchSimCalc` from a saved model.

        .. note::
            This method is private and intended for internal testing only.
            Users should use ``from_compiled_model`` instead.

        Args:
            model_path (str): path to a checkpoint file, package file, or ``nequip.net`` model ID.
            device (str or :class:`torch.device`): the device to use (e.g., ``"cpu"`` or ``"cuda"``).
            chemical_species_to_atom_type_map (Dict[str, str] or bool or None): mapping from chemical species to model type names.
                If ``None`` (default), uses identity mapping with warning.
                If ``True``, uses identity mapping without warning.
                If dict, uses the provided mapping.
            **kwargs: additional arguments passed to :class:`~nequip.integrations.torchsim.NequIPTorchSimCalc`.
        """
        from nequip.model.saved_models.load_utils import load_saved_model
        from nequip.train.lightning import _SOLE_MODEL_KEY

        # load model using unified loader
        model: graph_model.GraphModel = load_saved_model(
            model_path, model_key=_SOLE_MODEL_KEY
        )
        model.eval()

        r_max = float(model.metadata[graph_model.R_MAX_KEY])
        type_names = model.metadata[graph_model.TYPE_NAMES_KEY].split(" ")

        # handle chemical species mapping
        chemical_species_to_atom_type_map = cls._handle_chemical_species_map(
            chemical_species_to_atom_type_map, type_names
        )

        # check for transforms in kwargs
        if "transforms" in kwargs:
            raise KeyError("`transforms` not allowed here")

        return cls(
            model=model,
            r_max=r_max,
            device=device,
            transforms=_basic_transforms(
                model.metadata, r_max, type_names, chemical_species_to_atom_type_map
            ),
            **kwargs,
        )

    def setup_from_system_idx(
        self, atomic_numbers: torch.Tensor, system_idx: torch.Tensor
    ) -> None:
        """Set up internal state from atomic numbers and system indices.

        Args:
            atomic_numbers (:class:`torch.Tensor`): atomic numbers with shape ``[n_atoms]``.
            system_idx (:class:`torch.Tensor`): system indices with shape ``[n_atoms]``.
        """
        self.atomic_numbers = atomic_numbers
        self.system_idx = system_idx

        # determine number of systems and atoms per system
        self.n_systems = system_idx.max().item() + 1
        self.total_atoms = atomic_numbers.shape[0]

    def forward(self, state: ts.SimState | StateDict) -> dict[str, torch.Tensor]:  # noqa: C901
        """Compute energies, forces, and stresses.

        Args:
            state (:class:`~torch_sim.SimState` | :class:`~torch_sim.typing.StateDict`): state object containing positions, cell,
                and system information.

        Returns:
            dict[str, :class:`torch.Tensor`]: computed properties (``"energy"``, ``"forces"``, ``"stress"``).
        """
        sim_state = (
            state
            if isinstance(state, ts.SimState)
            else ts.SimState(**state, masses=torch.ones_like(state["positions"]))
        )

        # handle input validation for atomic numbers
        if sim_state.atomic_numbers is None and not self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers must be provided in either the constructor or forward."
            )
        if sim_state.atomic_numbers is not None and self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers cannot be provided in both the constructor and forward."
            )

        # use system_idx from init if not provided
        if sim_state.system_idx is None:
            if not hasattr(self, "system_idx"):
                raise ValueError(
                    "System indices must be provided if not set during initialization"
                )
            sim_state.system_idx = self.system_idx

        # update batch information if new atomic numbers are provided
        if (
            sim_state.atomic_numbers is not None
            and not self.atomic_numbers_in_init
            and not torch.equal(
                sim_state.atomic_numbers,
                getattr(self, "atomic_numbers", torch.zeros(0, device=self._device)),
            )
        ):
            self.setup_from_system_idx(sim_state.atomic_numbers, sim_state.system_idx)

        # === TS neighborlist construction ===
        edge_indices = []
        shifts_list = []
        unit_shifts_list = []
        offset = 0

        # TODO (AG): currently doesn't work for batched neighbor lists
        for sys_idx in range(self.n_systems):
            system_idx_mask = sim_state.system_idx == sys_idx
            # calculate neighbor list for this system
            edge_idx, shifts_idx = self.neighbor_list_fn(
                positions=sim_state.positions[system_idx_mask],
                cell=sim_state.row_vector_cell[sys_idx],
                pbc=sim_state.pbc,
                cutoff=self.r_max,
            )

            # adjust indices for the batch
            edge_idx = edge_idx + offset
            shifts = torch.mm(shifts_idx, sim_state.row_vector_cell[sys_idx])

            edge_indices.append(edge_idx)
            unit_shifts_list.append(shifts_idx)
            shifts_list.append(shifts)

            offset += len(sim_state.positions[system_idx_mask])

        # combine all neighbor lists
        edge_index = torch.cat(edge_indices, dim=1)
        unit_shifts = torch.cat(unit_shifts_list, dim=0)
        shifts = torch.cat(shifts_list, dim=0)

        # === prepare raw dict ===
        data: dict[str, torch.Tensor] = {
            AtomicDataDict.POSITIONS_KEY: sim_state.positions,
            AtomicDataDict.CELL_KEY: sim_state.row_vector_cell,
            AtomicDataDict.BATCH_KEY: sim_state.system_idx,
            AtomicDataDict.NUM_NODES_KEY: sim_state.system_idx.bincount(),
            AtomicDataDict.ATOMIC_NUMBERS_KEY: sim_state.atomic_numbers,
            AtomicDataDict.EDGE_INDEX_KEY: edge_index,
            AtomicDataDict.EDGE_CELL_SHIFT_KEY: unit_shifts,
        }

        # === apply transforms ===
        for t in self.transforms:
            data = t(data)

        # === run model ===
        out = self.model(data)

        # === collect outputs ===
        results: dict[str, torch.Tensor] = {}

        energy = out[AtomicDataDict.TOTAL_ENERGY_KEY]
        if energy is not None:
            results["energy"] = energy.view(-1).detach()
        else:
            results["energy"] = torch.zeros(self.n_systems, device=self._device)

        if self.compute_forces:
            forces = out[AtomicDataDict.FORCE_KEY]
            if forces is not None:
                results["forces"] = forces.detach()

        if self.compute_stress:
            stress = out[AtomicDataDict.STRESS_KEY]
            if stress is not None:
                results["stress"] = stress.detach()

        return results


def _basic_transforms(
    metadata: dict,
    r_max: float,
    type_names: List[str],
    chemical_species_to_atom_type_map: Dict[str, str],
) -> List[Callable]:
    """Create transform list with optional pruning based on per-edge-type cutoffs."""
    from nequip.data.transforms import (
        ChemicalSpeciesToAtomTypeMapper,
        NeighborListPruneTransform,
    )
    from nequip.nn.embedding.utils import cutoff_str_to_fulldict

    transforms = [
        ChemicalSpeciesToAtomTypeMapper(
            model_type_names=type_names,
            chemical_species_to_atom_type_map=chemical_species_to_atom_type_map,
        )
    ]

    # add pruning transform if per-edge-type cutoffs are available
    if metadata.get(graph_model.PER_EDGE_TYPE_CUTOFF_KEY, None) is not None:
        per_edge_type_cutoff = metadata[graph_model.PER_EDGE_TYPE_CUTOFF_KEY]
        if isinstance(per_edge_type_cutoff, str):
            per_edge_type_cutoff = cutoff_str_to_fulldict(
                per_edge_type_cutoff, type_names
            )

        transforms.append(
            NeighborListPruneTransform(
                r_max=r_max,
                per_edge_type_cutoff=per_edge_type_cutoff,
                type_names=type_names,
            )
        )

    return transforms
