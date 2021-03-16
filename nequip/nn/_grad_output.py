import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


@compile_mode("script")
class GradientOutput(GraphModuleMixin, torch.nn.Module):
    sign: float

    def __init__(
        self, func, of, wrt, out_field=None, irreps_in=None, sign: float = 1.0
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
        # Check irreps
        irreps_of = Irreps(irreps_in[of])
        if irreps_of.lmax > 0 or irreps_of.num_irreps > 1:
            raise NotImplementedError(
                "Currently, GradientOutput only supports taking gradients of single scalar outputs"
            )

        self._init_irreps(
            irreps_in=irreps_in,
        )
        # The gradient of a single scalar w.r.t. something of a given shape and irrep just has that shape and irrep
        # Ex.: gradient of energy (0e) w.r.t. position vector (L=1) is also an L = 1 vector
        self.irreps_out.update(
            {f: self.irreps_in[wrt] for f, wrt in zip(self.out_field, self.wrt)}
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # set req grad
        wrt_tensors = []
        for k in self.wrt:
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
            create_graph=True,  # needed to allow gradients of this output
        )
        # return
        # grad is optional[tensor]?
        for out, grad in zip(self.out_field, grads):
            if grad is None:
                # From the docs: "If an output doesn’t require_grad, then the gradient can be None"
                raise RuntimeError("Something is wrong, gradient couldn't be computed")
            else:
                data[out] = self.sign * grad

        return data
