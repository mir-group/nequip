# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import List, Union, Optional

import torch

from e3nn.o3._irreps import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin


@compile_mode("script")
class GradientOutput(GraphModuleMixin, torch.nn.Module):
    r"""Wrap a model and include as an output its gradient.

    Args:
        func: the model to wrap
        of: the name of the output field of ``func`` to take the gradient with respect to. The field must be a single scalar (i.e. have irreps ``0e``)
        wrt: the input field(s) of ``func`` to take the gradient of ``of`` with regards to.
        out_field: the field in which to return the computed gradients. Defaults to ``f"d({of})/d({wrt})"`` for each field in ``wrt``.
        sign: either 1 or -1; the returned gradient is multiplied by this.
    """

    sign: float
    _negate: bool
    skip: bool

    def __init__(
        self,
        func: GraphModuleMixin,
        of: str,
        wrt: Union[str, List[str]],
        out_field: Optional[List[str]] = None,
        sign: float = 1.0,
    ):
        super().__init__()
        sign = float(sign)
        assert sign in (1.0, -1.0)
        self.sign = sign
        self._negate = sign == -1.0
        self.of = of
        self.skip = False

        # TO DO: maybe better to force using list?
        if isinstance(wrt, str):
            wrt = [wrt]
        if isinstance(out_field, str):
            out_field = [out_field]
        self.wrt = wrt
        self.func = func
        if out_field is None:
            self.out_field = [f"d({of})/d({e})" for e in self.wrt]
        else:
            assert len(out_field) == len(
                self.wrt
            ), "Out field names must be given for all w.r.t tensors"
            self.out_field = out_field

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={of: Irreps("0e")},
            irreps_out=func.irreps_out,
        )

        # The gradient of a single scalar w.r.t. something of a given shape and irrep just has that shape and irrep
        # Ex.: gradient of energy (0e) w.r.t. position vector (L=1) is also an L = 1 vector
        self.irreps_out.update(
            {f: self.irreps_in[wrt] for f, wrt in zip(self.out_field, self.wrt)}
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        if self.skip:
            return self.func(data)

        # set req grad
        wrt_tensors = []
        old_requires_grad: List[bool] = []
        for k in self.wrt:
            old_requires_grad.append(data[k].requires_grad)
            data[k].requires_grad_(True)
            wrt_tensors.append(data[k])
        # run func
        data = self.func(data)
        # Get grads
        grads = torch.autograd.grad(
            # TODO:
            # This makes sense for scalar batch-level or batch-wise outputs, specifically because d(sum(batches))/d wrt = sum(d batch / d wrt) = d my_batch / d wrt
            # for a well-behaved example level like energy where d other_batch / d wrt is always zero. (In other words, the energy of example 1 in the batch is completely unaffect by changes in the position of atoms in another example.)
            # This should work for any gradient of energy, but could act suspiciously and unexpectedly for arbitrary gradient outputs, if they ever come up
            [data[self.of].sum()],
            wrt_tensors,
            create_graph=self.training,  # needed to allow gradients of this output during training
        )
        # return
        # grad is optional[tensor]?
        for out, grad in zip(self.out_field, grads):
            if grad is None:
                # From the docs: "If an output doesn’t require_grad, then the gradient can be None"
                raise RuntimeError("Something is wrong, gradient couldn't be computed")

            if self._negate:
                grad = torch.neg(grad)
            data[out] = grad

        # unset requires_grad_
        for req_grad, k in zip(old_requires_grad, self.wrt):
            data[k].requires_grad_(req_grad)

        return data


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
            # [natom, 3] @ [3, 3] -> [natom, 3]
            data[AtomicDataDict.POSITIONS_KEY] = torch.addmm(
                pos, pos, symmetric_displacement
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
                tmpcell = cell.squeeze(0)
                data[AtomicDataDict.CELL_KEY] = torch.addmm(
                    tmpcell, tmpcell, symmetric_displacement
                ).unsqueeze(0)
            # assert torch.equal(cell, data[AtomicDataDict.CELL_KEY])

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
