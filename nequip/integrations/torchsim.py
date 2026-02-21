"""Wrapper for NequIP framework models in torch-sim"""

import torch

import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.typing import StateDict

from nequip.data import AtomicDataDict

from .mixins import _IntegrationLoaderMixin

from collections.abc import Callable
from typing import Union, List, Dict


class NequIPTorchSimCalc(_IntegrationLoaderMixin, ModelInterface):
    """NequIP framework torch-sim calculator.

    This torch-sim calculator is compatible with models from the NequIP framework,
    including NequIP and Allegro models.

    The recommended way to use this calculator is with a compiled model, i.e.
    ``nequip-compile`` the model and load it into the calculator with
    ``NequIPTorchSimCalc.from_compiled_model(...)``.

    Args:
        model (:class:`torch.nn.Module`): a model in the NequIP framework
        device (str or :class:`torch.device`): device for model to evaluate on,
            e.g. ``"cpu"`` or ``"cuda"`` (default: ``"cpu"``)
        transforms (List[Callable]): list of data transforms
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
        device: Union[str, torch.device] = "cpu",
        transforms: List[Callable] = [],
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

        if not isinstance(model, torch.nn.Module):
            raise TypeError("Invalid model type. Must be a torch.nn.Module.")
        self.model = model.to(self._device)

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
    def _get_aoti_compile_target(cls) -> Dict:
        from nequip.scripts._compile_utils import COMPILE_TARGET_DICT, AOTI_BATCH_TARGET

        return COMPILE_TARGET_DICT[AOTI_BATCH_TARGET]

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

        # === prepare raw dict ===
        # convert PBC to tensor with shape [n_systems, 3] for batched data
        pbc = sim_state.pbc
        # the following logic accounts for torch-sim change:
        # https://github.com/TorchSim/torch-sim/pull/320
        if isinstance(pbc, bool):
            # previously, pbc is a bool
            pbc = torch.tensor([pbc] * 3, dtype=torch.bool, device=self._device)
        # after PR, pbc is already a tensor with shape [3]
        # expand to [n_systems, 3] for batched processing
        pbc_tensor = pbc.unsqueeze(0).expand(self.n_systems, 3)

        data: dict[str, torch.Tensor] = {
            AtomicDataDict.POSITIONS_KEY: sim_state.positions,
            AtomicDataDict.CELL_KEY: sim_state.row_vector_cell,
            AtomicDataDict.PBC_KEY: pbc_tensor,
            AtomicDataDict.BATCH_KEY: sim_state.system_idx,
            AtomicDataDict.NUM_NODES_KEY: sim_state.system_idx.bincount(),
            AtomicDataDict.ATOMIC_NUMBERS_KEY: sim_state.atomic_numbers,
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

        self.save_extra_outputs(out, results)

        return results

    def save_extra_outputs(
        self, out: dict[str, torch.Tensor], results: dict[str, torch.Tensor]
    ) -> None:
        # subclasses can implement this method to process extra outputs without code duplication
        pass
