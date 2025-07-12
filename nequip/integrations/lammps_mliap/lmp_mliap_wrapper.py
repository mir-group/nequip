# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import torch

from nequip.data import AtomicDataDict
from nequip.nn import graph_model
from nequip.model.saved_models.load_utils import load_saved_model
from nequip.model.modify_utils import get_all_modifiers, modify
from nequip.model.utils import _EAGER_MODEL_KEY
from nequip.utils.global_state import set_global_state
from nequip.utils.compile import prepare_model_for_compile
from nequip.utils.versions import _TORCH_GE_2_6

from lammps.mliap.mliap_unified_abc import MLIAPUnified

from typing import List


class NequIPLAMMPSMLIAPWrapper(MLIAPUnified):
    """LAMMPS-MLIAP interface for NequIP framework models."""

    model_path: str

    def __init__(
        self,
        model_path: str,
        model_key: str,
        modifiers: List[str] = [],
        compile: bool = True,
        tf32: bool = False,
        **kwargs,
    ):
        # this is a white lie, unsure if strictly necessary, but just in case
        assert (
            _TORCH_GE_2_6
        ), "PyTorch >= 2.6 required for NequIP's LAMMPS ML-IAP interface"
        super().__init__()
        self.model_path = model_path
        self.model_key = model_key
        self.modifiers = modifiers
        self.compile = compile
        self.tf32 = tf32
        self.model = None
        self.device = None
        self.type_mapper = None

        # to placate the interface
        self.nparams = 1
        self.ndescriptors = 1

        # === set model-depnedent params ===
        set_global_state()
        model = load_saved_model(
            self.model_path,
            compile_mode=_EAGER_MODEL_KEY,
            model_key=self.model_key,
        )
        self.rcutfac = 0.5 * float(model.metadata[graph_model.R_MAX_KEY])
        # TODO: we are assuming model type names are element names here
        # but this might not be true
        # but a fundamental change is required to be more flexible
        self.element_types = model.type_names.copy()

    def _initialize_model(self, lmp_data) -> None:
        # initialize global state
        set_global_state(allow_tf32=self.tf32)
        # load eager model
        model = load_saved_model(
            self.model_path,
            compile_mode=_EAGER_MODEL_KEY,
            model_key=self.model_key,
        )

        # apply LAMMPS MLIAP ghost exchange modifier if present
        available_modifiers = get_all_modifiers(model)
        _HAS_MLIAP_GHOST_EXCHANGE = (
            "enable_LAMMPSMLIAPGhostExchange" in available_modifiers
        )
        if _HAS_MLIAP_GHOST_EXCHANGE:
            model = modify(model, [{"modifier": "enable_LAMMPSMLIAPGhostExchange"}])

        # NOTE: this is a hack/workaround because modifiers like NequIP-OEQ condition modification logic around whether `torch.compile` will be called
        # even though we load an eager model, we just set this attribute to True for the acceleration modifiers
        # and reset to False after modification
        model.is_compile_graph_model = True

        # apply other modifiers
        if self.modifiers:
            model = modify(
                model, [{"modifier": modifier} for modifier in self.modifiers]
            )

        model.is_compile_graph_model = False

        # set device and "compile" model
        self.device = (
            "cuda" if "kokkos" in lmp_data.__class__.__module__.lower() else "cpu"
        )
        model = prepare_model_for_compile(model, self.device)

        # make the model an energy model so that we can rely on AOT Autograd for inference
        # model is `GraphModel(StressForceOutput(EnergyModel))`
        # we have to do it this way since `torch.compile` can't handle `x.requires_grad_(True)`
        # we avoid using the `CompileGraphModel` because of potential batch dim issues, potential make_fx issues with the ghost exchange module, and because it was written specifically for train-time compile
        model.model = model.model.func

        if self.compile:
            self.model = torch.compile(
                model, dynamic=True, fullgraph=not _HAS_MLIAP_GHOST_EXCHANGE
            )
        else:
            self.model = model

    def compute_forces(self, lmp_data):

        # === lazily load model ===
        if self.model is None:
            self._initialize_model(lmp_data)

        if lmp_data.nlocal == 0 or lmp_data.npairs <= 1:
            return

        # === create input data ===
        # make sure edge vectors `requires_grad`
        edge_vectors = torch.as_tensor(lmp_data.rij, dtype=torch.float64).to(
            self.device
        )
        edge_vectors.requires_grad_(True)
        nequip_data_in = {
            AtomicDataDict.EDGE_VECTORS_KEY: edge_vectors,
            AtomicDataDict.EDGE_INDEX_KEY: torch.vstack(
                [
                    torch.as_tensor(lmp_data.pair_i, dtype=torch.int64).to(self.device),
                    torch.as_tensor(lmp_data.pair_j, dtype=torch.int64).to(self.device),
                ],
            ),
            AtomicDataDict.ATOM_TYPE_KEY: torch.as_tensor(
                lmp_data.elems, dtype=torch.int64
            ).to(self.device),
            AtomicDataDict.LMP_MLIAP_DATA_KEY: lmp_data,
        }

        # === run model ===
        # run model and backwards for edge forces
        nequip_data_in[AtomicDataDict.EDGE_VECTORS_KEY].requires_grad_(True)
        nequip_data_out = self.model(nequip_data_in)
        # correct sign convention for consistency with LAMMPS
        edge_forces = torch.autograd.grad(
            [nequip_data_out[AtomicDataDict.TOTAL_ENERGY_KEY].sum()],
            [edge_vectors],
        )[0]

        # === pass outputs to LAMMPS ===
        # handle ghosts
        nequip_atomic_energies = nequip_data_out[
            AtomicDataDict.PER_ATOM_ENERGY_KEY
        ].view(-1)

        # shape-dependent control flow, but should be outside of compiled model
        if nequip_atomic_energies.size(0) != lmp_data.nlocal:
            # per-atom quantities that come out of the model would be on `num_local + num_ghost`
            nequip_atomic_energies = torch.narrow(
                nequip_atomic_energies, 0, 0, lmp_data.nlocal
            )
            nequip_total_energy = torch.sum(nequip_atomic_energies)
        else:
            nequip_total_energy = nequip_data_out[AtomicDataDict.TOTAL_ENERGY_KEY]

        # update LAMMPS variables
        lmp_eatoms = torch.as_tensor(lmp_data.eatoms)
        lmp_eatoms.copy_(nequip_atomic_energies)
        lmp_data.energy = nequip_total_energy
        lmp_data.update_pair_forces_gpu(edge_forces)

    def compute_descriptors(self, lmp_data):
        pass

    def compute_gradients(self, lmp_data):
        pass
