from typing import List, Union, Optional

import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


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
        self.of = of
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
            else:
                data[out] = self.sign * grad

        # unset requires_grad_
        for req_grad, k in zip(old_requires_grad, self.wrt):
            data[k].requires_grad_(req_grad)

        return data


@compile_mode("unsupported")
class PartialForceOutput(GraphModuleMixin, torch.nn.Module):
    r"""Wrap a model and include as an output its gradient."""
    vectorize: bool

    def __init__(
        self,
        func: GraphModuleMixin,
        vectorize: bool = False,
    ):
        super().__init__()
        # TODO wrap:
        self.func = func
        self.vectorize = vectorize
        if vectorize:
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
