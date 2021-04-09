from typing import Union

import torch
from e3nn import o3
from e3nn.util.test import assert_equivariant

from nequip.nn import GraphModuleMixin
from nequip.data import AtomicData, AtomicDataDict


def assert_AtomicData_equivariant(
    func: GraphModuleMixin, data_in: Union[AtomicData, AtomicDataDict.Type], **kwargs
):
    # Prevent pytest from showing this function in the traceback
    __tracebackhide__ = True

    irreps_in = {k: None for k in AtomicDataDict.ALLOWED_KEYS}
    irreps_in.update(
        {
            AtomicDataDict.POSITIONS_KEY: "cartesian_points",
            AtomicDataDict.CELL_KEY: "3x1o",
        }
    )
    irreps_in.update(func.irreps_in)
    irreps_in = {k: v for k, v in irreps_in.items() if k in data_in}

    def wrapper(*args):
        arg_dict = {k: v for k, v in zip(irreps_in, args)}
        # cell is a special case
        if AtomicDataDict.CELL_KEY in arg_dict:
            # unflatten
            cell = arg_dict[AtomicDataDict.CELL_KEY]
            assert cell.shape[-1] == 9
            arg_dict[AtomicDataDict.CELL_KEY] = cell.reshape(cell.shape[:-1] + (3, 3))
        output = func(arg_dict)
        # cell is a special case
        if AtomicDataDict.CELL_KEY in output:
            # flatten
            cell = arg_dict[AtomicDataDict.CELL_KEY]
            assert cell.shape[-2:] == (3, 3)
            arg_dict[AtomicDataDict.CELL_KEY] = cell.reshape(cell.shape[:-2] + (9,))
        return [output[k] for k in func.irreps_out]

    data_in = AtomicData.to_AtomicDataDict(data_in)
    # cell is a special case
    if AtomicDataDict.CELL_KEY in data_in:
        # flatten
        cell = data_in[AtomicDataDict.CELL_KEY]
        assert cell.shape[-2:] == (3, 3)
        data_in[AtomicDataDict.CELL_KEY] = cell.reshape(cell.shape[:-2] + (9,))

    args_in = [data_in[k] for k in irreps_in]

    return assert_equivariant(
        wrapper,
        args_in=args_in,
        irreps_in=list(irreps_in.values()),
        irreps_out=list(func.irreps_out.values()),
        **kwargs,
    )


_DEBUG_HOOKS = None


def set_irreps_debug(enabled: bool = False):
    """Add debugging hooks to ``forward()`` that check data-irreps consistancy."""
    global _DEBUG_HOOKS
    if _DEBUG_HOOKS is None and not enabled:
        return
    elif _DEBUG_HOOKS is not None and enabled:
        return
    elif _DEBUG_HOOKS is not None and not enabled:
        for hook in _DEBUG_HOOKS:
            hook.remove()
        _DEBUG_HOOKS = None
        return
    else:
        pass

    import torch.nn.modules
    from torch_geometric.data import Data

    def pre_hook(mod: GraphModuleMixin, inp):
        __tracebackhide__ = True
        if not isinstance(mod, GraphModuleMixin):
            return
        mname = type(mod).__name__
        if len(inp) > 1:
            raise ValueError(
                f"Module {mname} should have received a single argument, but got {len(inp)}"
            )
        elif len(inp) == 0:
            raise ValueError(
                f"Module {mname} didn't get any arguments; this case is correctly handled with an empty dict."
            )
        inp = inp[0]
        if not (isinstance(inp, dict) or isinstance(inp, Data)):
            raise TypeError(
                f"Module {mname} should have received a dict or a torch_geometric Data, instead got a {type(inp).__name__}"
            )
        for k, ir in mod.irreps_in.items():
            if k not in inp:
                raise KeyError(
                    f"Field {k} with irreps {ir} expected to be input to {mname}; not present"
                )
            elif isinstance(inp[k], torch.Tensor) and isinstance(ir, o3.Irreps):
                if inp[k].ndim == 1:
                    raise ValueError(
                        f"Field {k} in input to module {mname} has only one dimension (assumed to be batch-like); it must have a second irreps dimension even if irreps.dim == 1 (i.e. a single per atom scalar must have shape [N_at, 1], not [N_at])"
                    )
                elif inp[k].shape[-1] != ir.dim:
                    raise ValueError(
                        f"Field {k} in input to module {mname} has last dimension {inp[k].shape[-1]} but its irreps {ir} indicate last dimension {ir.dim}"
                    )
        return

    h1 = torch.nn.modules.module.register_module_forward_pre_hook(pre_hook)

    def post_hook(mod: GraphModuleMixin, _, out):
        __tracebackhide__ = True
        if not isinstance(mod, GraphModuleMixin):
            return
        mname = type(mod).__name__
        if not (isinstance(out, dict) or isinstance(out, Data)):
            raise TypeError(
                f"Module {mname} should have returned a dict or a torch_geometric Data, instead got a {type(out).__name__}"
            )
        for k, ir in mod.irreps_out.items():
            if k not in out:
                raise KeyError(
                    f"Field {k} with irreps {ir} expected to be in output from {mname}; not present"
                )
            elif isinstance(out[k], torch.Tensor) and isinstance(ir, o3.Irreps):
                if out[k].ndim == 1:
                    raise ValueError(
                        f"Field {k} in output from module {mname} has only one dimension (assumed to be batch-like); it must have a second irreps dimension even if irreps.dim == 1 (i.e. a single per atom scalar must have shape [N_at, 1], not [N_at])"
                    )
                elif out[k].shape[-1] != ir.dim:
                    raise ValueError(
                        f"Field {k} in output from {mname} has last dimension {out[k].shape[-1]} but its irreps {ir} indicate last dimension {ir.dim}"
                    )
        return

    h2 = torch.nn.modules.module.register_module_forward_hook(post_hook)

    _DEBUG_HOOKS = (h1, h2)
    return
