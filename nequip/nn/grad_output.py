# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import torch

from e3nn.o3._irreps import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin


@compile_mode("unsupported")
class PartialForceOutput(GraphModuleMixin, torch.nn.Module):
    r"""Generate partial and total forces from an energy model.

    Args:
        func: the energy model
        vectorize: the vectorize option to ``torch.autograd.functional.jacobian``,
            false by default since it doesn't work well.
    """

    vectorize: bool

    def __init__(
        self,
        func: GraphModuleMixin,
        vectorize: bool = False,
        vectorize_warnings: bool = False,
    ):
        super().__init__()
        self.func = func
        self.vectorize = vectorize
        if vectorize_warnings:
            # See https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html
            torch._C._debug_only_display_vmap_fallback_warnings(True)

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={AtomicDataDict.PER_ATOM_ENERGY_KEY: Irreps("0e")},
            irreps_out=func.irreps_out,
        )
        self.irreps_out[AtomicDataDict.PARTIAL_FORCE_KEY] = Irreps("1o")
        self.irreps_out[AtomicDataDict.FORCE_KEY] = Irreps("1o")

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = data.copy()
        out_data = {}

        def wrapper(pos: torch.Tensor) -> torch.Tensor:
            """Wrapper from pos to atomic energy"""
            nonlocal data, out_data
            data[AtomicDataDict.POSITIONS_KEY] = pos
            out_data = self.func(data)
            return out_data[AtomicDataDict.PER_ATOM_ENERGY_KEY].squeeze(-1)

        pos = data[AtomicDataDict.POSITIONS_KEY]

        partial_forces = torch.autograd.functional.jacobian(
            func=wrapper,
            inputs=pos,
            create_graph=self.training,  # needed to allow gradients of this output during training
            vectorize=self.vectorize,
        )
        partial_forces = partial_forces.negative()
        # output is [n_at, n_at, 3]

        out_data[AtomicDataDict.PARTIAL_FORCE_KEY] = partial_forces
        out_data[AtomicDataDict.FORCE_KEY] = partial_forces.sum(dim=0)

        return out_data


@compile_mode("script")
class ForceStressOutput(GraphModuleMixin, torch.nn.Module):
    r"""Compute forces (and stress if cell is provided) using autograd of an energy model.

    See:
        Knuth et. al. Comput. Phys. Commun 190, 33-50, 2015
        https://pure.mpg.de/rest/items/item_2085135_9/component/file_2156800/content

    Args:
        func: the energy model to wrap
    """

    def __init__(self, func: GraphModuleMixin):
        super().__init__()
        self.func = func

        # check and init irreps
        self._init_irreps(
            irreps_in=self.func.irreps_in.copy(),
            irreps_out=self.func.irreps_out.copy(),
        )
        self.irreps_out[AtomicDataDict.FORCE_KEY] = "1o"
        self.irreps_out[AtomicDataDict.STRESS_KEY] = "1o"
        self.irreps_out[AtomicDataDict.VIRIAL_KEY] = "1o"

        # for torchscript compat
        self.register_buffer("_empty", torch.Tensor())

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        assert AtomicDataDict.EDGE_VECTORS_KEY not in data

        if AtomicDataDict.BATCH_KEY in data:
            batch = data[AtomicDataDict.BATCH_KEY]
            num_batch: int = AtomicDataDict.num_frames(data)
        else:
            # Special case for efficiency
            batch = self._empty
            num_batch: int = 1

        pos = data[AtomicDataDict.POSITIONS_KEY]
        has_cell: bool = AtomicDataDict.CELL_KEY in data

        if has_cell:
            orig_cell = data[AtomicDataDict.CELL_KEY]
            # Make the cell per-batch
            cell = orig_cell.view(-1, 3, 3).expand(num_batch, 3, 3)
            data[AtomicDataDict.CELL_KEY] = cell
        else:
            # torchscript
            orig_cell = self._empty
            cell = self._empty
        # Add the displacements
        # the GradientOutput will make them require grad
        # See SchNetPack code:
        # https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/atomistic/model.py#L45
        # SchNetPack issue:
        # https://github.com/atomistic-machine-learning/schnetpack/issues/165
        # Paper they worked from:
        # Knuth et. al. Comput. Phys. Commun 190, 33-50, 2015
        # https://pure.mpg.de/rest/items/item_2085135_9/component/file_2156800/content

        if num_batch > 1:
            displacement = torch.zeros(
                (num_batch, 3, 3),
                dtype=pos.dtype,
                device=pos.device,
            )
        else:
            displacement = torch.zeros(
                (3, 3),
                dtype=pos.dtype,
                device=pos.device,
            )
        displacement.requires_grad_(True)
        data["_displacement"] = displacement
        # in the above paper, the infinitesimal distortion is *symmetric*
        # so we symmetrize the displacement before applying it to
        # the positions/cell
        # This is not strictly necessary (reasoning thanks to Mario):
        # the displacement's asymmetric 1o term corresponds to an
        # infinitesimal rotation, which should not affect the final
        # output (invariance).
        # That said, due to numerical error, this will never be
        # exactly true. So, we symmetrize the deformation to
        # take advantage of this understanding and not rely on
        # the invariance here:
        symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
        did_pos_req_grad: bool = pos.requires_grad
        pos.requires_grad_(True)
        if num_batch > 1:
            # bmm is natom in batch
            # batched [natom, 1, 3] @ [natom, 3, 3] -> [natom, 1, 3] -> [natom, 3]
            data[AtomicDataDict.POSITIONS_KEY] = pos + torch.bmm(
                pos.unsqueeze(-2), torch.index_select(symmetric_displacement, 0, batch)
            ).squeeze(-2)
        else:
            # (num_atoms, 3), (3, 3) -> (num_atoms, 3)
            data[AtomicDataDict.POSITIONS_KEY] = pos + torch.sum(
                pos.view(-1, 3, 1) * symmetric_displacement, 1
            )
        # assert torch.equal(pos, data[AtomicDataDict.POSITIONS_KEY])
        # we only displace the cell if we have one:
        if has_cell:
            # bmm is num_batch in batch
            # here we apply the distortion to the cell as well
            # this is critical also for the correctness
            # if we didn't symmetrize the distortion, since without this
            # there would then be an infinitesimal rotation of the positions
            # but not cell, and it thus wouldn't be global and have
            # no effect due to equivariance/invariance.
            if num_batch > 1:
                # [n_batch, 3, 3] @ [n_batch, 3, 3]
                data[AtomicDataDict.CELL_KEY] = cell + torch.bmm(
                    cell, symmetric_displacement
                )
            else:
                # [3, 3] @ [3, 3] --- enforced to these shapes
                data[AtomicDataDict.CELL_KEY] = (
                    cell.view(3, 3)
                    + torch.sum(cell.view(3, 3, 1) * symmetric_displacement, 1)
                ).view(1, 3, 3)

        # Call model and get gradients
        data = self.func(data)

        grads = torch.autograd.grad(
            [data[AtomicDataDict.TOTAL_ENERGY_KEY].sum()],
            [pos, data["_displacement"]],
            create_graph=self.training,  # needed to allow gradients of this output during training
        )

        # Put negative sign on forces
        forces = grads[0]
        if forces is None:
            # condition needed to unwrap optional for torchscript
            assert False, "failed to compute forces autograd"
        forces = torch.neg(forces)
        data[AtomicDataDict.FORCE_KEY] = forces

        # Store virial
        virial = grads[1]
        if virial is None:
            # condition needed to unwrap optional for torchscript
            assert False, "failed to compute virial autograd"
        virial = virial.view(num_batch, 3, 3)

        # we only compute the stress (1/V * virial) if we have a cell whose volume we can compute
        if has_cell:
            # ^ can only scale by cell volume if we have one...:
            # Rescale stress tensor
            # See https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/atomistic/output_modules.py#L180
            # See also https://en.wikipedia.org/wiki/Triple_product
            # See also https://gitlab.com/ase/ase/-/blob/master/ase/cell.py,
            #          which uses np.abs(np.linalg.det(cell))
            # First dim is batch, second is vec, third is xyz
            # Note the .abs(), since volume should always be positive
            # det is equal to a dot (b cross c)
            volume = torch.linalg.det(cell).abs().unsqueeze(-1)
            stress = virial / volume.view(num_batch, 1, 1)
            data[AtomicDataDict.CELL_KEY] = orig_cell
        else:
            stress = self._empty  # torchscript
        data[AtomicDataDict.STRESS_KEY] = stress

        # see discussion in https://github.com/libAtoms/QUIP/issues/227 about sign convention
        # (and conventions docs page)
        # they say the standard convention is virial = -stress x volume
        # looking above this means that we need to pick up another negative sign for the virial
        # to fit this equation with the stress computed above
        virial = torch.neg(virial)
        data[AtomicDataDict.VIRIAL_KEY] = virial

        # Remove helper
        del data["_displacement"]
        if not did_pos_req_grad:
            # don't give later modules one that does
            pos.requires_grad_(False)

        return data
