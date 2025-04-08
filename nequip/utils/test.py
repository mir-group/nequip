# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import ase
import torch
from e3nn import o3
from e3nn.util.test import equivariance_error

from nequip.nn import GraphModuleMixin, GraphModel
from nequip.data import (
    from_dict,
    from_ase,
    compute_neighborlist_,
    AtomicDataDict,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _CARTESIAN_TENSOR_FIELDS,
)
from .dtype import dtype_from_name

from functools import wraps
from typing import Union, Optional, List


# The default float tolerance
FLOAT_TOLERANCE = {
    t: torch.as_tensor(v, dtype=dtype_from_name(t))
    for t, v in {"float32": 1e-3, "float64": 1e-10}.items()
}
# Allow lookup by name or dtype object:
for t, v in list(FLOAT_TOLERANCE.items()):
    FLOAT_TOLERANCE[dtype_from_name(t)] = v
del t, v

# This has to be somewhat large because of float32 sum reductions over many edges/atoms
PERMUTATION_FLOAT_TOLERANCE = {torch.float32: 1e-4, torch.float64: 1e-10}


# https://discuss.pytorch.org/t/how-to-quickly-inverse-a-permutation-by-using-pytorch/116205/4
def _inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


def assert_permutation_equivariant(
    func: GraphModuleMixin,
    data_in: AtomicDataDict.Type,
    tolerance: Optional[float] = None,
    raise_error: bool = True,
) -> str:
    r"""Test the permutation equivariance of ``func``.

    Standard fields are assumed to be equivariant to node or edge permutations according to their standard interpretions; all other fields are assumed to be invariant to all permutations. Non-standard fields can be registered as node/edge permutation equivariant using ``register_fields``.

    Raises ``AssertionError`` if issues are found.

    Args:
        func: the module or model to test
        data_in: the example input data to test with
    """
    # Prevent pytest from showing this function in the traceback
    __tracebackhide__ = True

    if tolerance is None:
        atol = PERMUTATION_FLOAT_TOLERANCE[
            (
                func.model_dtype
                if isinstance(func, GraphModel)
                else torch.get_default_dtype()
            )
        ]
    else:
        atol = tolerance

    data_in = data_in.copy()
    device = data_in[AtomicDataDict.POSITIONS_KEY].device

    node_permute_fields = _NODE_FIELDS
    edge_permute_fields = _EDGE_FIELDS

    # Make permutations and make sure they are not identities
    n_node: int = len(data_in[AtomicDataDict.POSITIONS_KEY])
    while True:
        node_perm = torch.randperm(n_node, device=device)
        if n_node <= 1:
            break  # otherwise inf loop
        if not torch.all(node_perm == torch.arange(n_node, device=device)):
            break
    n_edge: int = data_in[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
    while True:
        edge_perm = torch.randperm(n_edge, device=device)
        if n_edge <= 1:
            break  # otherwise inf loop
        if not torch.all(edge_perm == torch.arange(n_edge, device=device)):
            break
    # ^ note that these permutations are maps from the "to" index to the "from" index
    # because we index by them, the 0th element of the permuted array will be the ith
    # of the original array, where i = perm[0]. Thus i is "from" and 0 is to, so perm
    # interpreted as a map is a map from "to" to "from".

    perm_data_in = {}
    for k in data_in.keys():
        if k in node_permute_fields:
            perm_data_in[k] = data_in[k][node_perm]
        elif k in edge_permute_fields:
            perm_data_in[k] = data_in[k][edge_perm]
        else:
            perm_data_in[k] = data_in[k]

    perm_data_in[AtomicDataDict.EDGE_INDEX_KEY] = _inverse_permutation(node_perm)[
        data_in[AtomicDataDict.EDGE_INDEX_KEY]
    ][:, edge_perm]

    out_orig = func(data_in)
    out_perm = func(perm_data_in)

    assert set(out_orig.keys()) == set(
        out_perm.keys()
    ), "Permutation changed the set of fields returned by model"

    messages = []
    num_problems: int = 0
    for k in out_orig.keys():
        if k in node_permute_fields:
            err = (out_orig[k][node_perm] - out_perm[k]).abs().max()
            fail = not torch.allclose(out_orig[k][node_perm], out_perm[k], atol=atol)
            if fail:
                num_problems += 1
            messages.append(
                f"   node permutation equivariance of field {k:22}       -> max error={err:.3e}{'  FAIL' if fail else ''}"
            )
        elif k in edge_permute_fields:
            err = (out_orig[k][edge_perm] - out_perm[k]).abs().max()
            fail = not torch.allclose(out_orig[k][edge_perm], out_perm[k], atol=atol)
            if fail:
                num_problems += 1
            messages.append(
                f"   edge permutation equivariance of field {k:22}       -> max error={err:.3e}{'  FAIL' if fail else ''}"
            )
        elif k in [AtomicDataDict.EDGE_INDEX_KEY, AtomicDataDict.EDGE_TYPE_KEY]:
            pass
        else:
            # Assume invariant
            if out_orig[k].dtype == torch.bool:
                err = (out_orig[k] != out_perm[k]).max()
            elif (out_orig[k].numel() == 0) and (out_perm[k].numel() == 0):
                err = 0.0
            else:
                err = (out_orig[k] - out_perm[k]).abs().max()
            fail = not torch.allclose(out_orig[k], out_perm[k], atol=atol)
            if fail:
                num_problems += 1
            messages.append(
                f"   edge & node permutation invariance for field {k:22} -> max error={err:.3e}{'  FAIL' if fail else ''}"
            )
    msg = "\n".join(messages)
    if num_problems == 0:
        return msg
    else:
        if raise_error:
            raise AssertionError(msg)
        else:
            return msg


def assert_AtomicData_equivariant(
    func: GraphModuleMixin,
    data_in: Union[AtomicDataDict.Type, List[AtomicDataDict.Type]],
    permutation_tolerance: Optional[float] = None,
    e3_tolerance: Optional[float] = None,
    **kwargs,
) -> str:
    r"""Test the rotation, translation, parity, and permutation equivariance of ``func``.

    For details on permutation testing, see ``assert_permutation_equivariant``.
    For details on geometric equivariance testing, see ``e3nn.util.test.assert_equivariant``.

    Raises ``AssertionError`` if issues are found.

    Args:
        func: the module or model to test
        data_in: the example input data(s) to test with. Only the first is used for permutation testing.
        **kwargs: passed to ``e3nn.util.test.assert_equivariant``

    Returns:
        A string description of the errors.
    """
    # Prevent pytest from showing this function in the traceback
    __tracebackhide__ = True

    if not isinstance(data_in, list):
        data_in = [data_in]
    data_in = [from_dict(d) for d in data_in]
    device, dtype = (
        data_in[0][AtomicDataDict.POSITIONS_KEY].device,
        data_in[0][AtomicDataDict.POSITIONS_KEY].dtype,
    )

    # == Test permutation of graph nodes ==
    # since permutation is discrete and should not be data dependent, run only on one frame.
    permutation_message = assert_permutation_equivariant(
        func, data_in[0], tolerance=permutation_tolerance, raise_error=False
    )

    # == Test rotation, parity, and translation using e3nn ==
    irreps_in = {k: None for k in AtomicDataDict.ALLOWED_KEYS}
    irreps_in.update(func.irreps_in)
    irreps_in = {k: v for k, v in irreps_in.items() if k in data_in[0]}
    irreps_out = func.irreps_out.copy()
    # Remove batch-related keys from the irreps_out, if we aren't using batched inputs
    irreps_out = {
        k: v
        for k, v in irreps_out.items()
        if not (k in ("batch", "ptr") and "batch" not in data_in)
    }
    # Assume that all data in `data_in` have the same keys
    # remove empty outputs from irreps out by running one set of data through
    data_in_0_copy = data_in[0].copy()
    ref_out = func(data_in_0_copy)
    for k, v in ref_out.items():
        if v.numel() == 0 and k in irreps_out.keys():
            _ = irreps_out.pop(k, None)

    # for certain things, we don't care what the given irreps are...
    # make sure that we test correctly for equivariance:
    for irps in (irreps_in, irreps_out):
        if AtomicDataDict.POSITIONS_KEY in irps:
            # it should always have been 1o vectors
            # since that's actually a valid Irreps
            assert o3.Irreps(irps[AtomicDataDict.POSITIONS_KEY]) == o3.Irreps("1o")
            irps[AtomicDataDict.POSITIONS_KEY] = "cartesian_points"
        if AtomicDataDict.CELL_KEY in irps:
            prev_cell_irps = irps[AtomicDataDict.CELL_KEY]
            assert prev_cell_irps is None or o3.Irreps(prev_cell_irps) == o3.Irreps(
                "1o"
            )
            # must be this to actually rotate it when flattened
            irps[AtomicDataDict.CELL_KEY] = "3x1o"

    cartesian_keys = _CARTESIAN_TENSOR_FIELDS.keys()
    for k in (
        AtomicDataDict.STRESS_KEY,
        AtomicDataDict.VIRIAL_KEY,
    ):  # TODO should this be cartesian_keys?
        irreps_in.pop(k, None)
    if any(k in irreps_out for k in cartesian_keys):
        from e3nn.io import CartesianTensor

        cartesian_tensor = {
            k: CartesianTensor(_CARTESIAN_TENSOR_FIELDS[k])
            for k in cartesian_keys
            if k in irreps_out
        }
        cartesian_rtp = {
            k: ct.reduced_tensor_products().to(device, dtype)
            for k, ct in cartesian_tensor.items()
        }
        for k, ct in cartesian_tensor.items():
            irreps_out[k] = ct

    def wrapper(*args):
        arg_dict = {k: v for k, v in zip(irreps_in, args)}
        # cell is a special case
        for key in (AtomicDataDict.CELL_KEY,):
            if key in arg_dict:
                # unflatten
                val = arg_dict[key]
                assert val.shape[-1] == 9
                arg_dict[key] = val.reshape(val.shape[:-1] + (3, 3))
        output = func(arg_dict)
        # irreps_out has been purged of numel=0 fields by now
        output = {k: output[k] for k, v in irreps_out.items() if k in output}
        # cell is a special case
        for key in (AtomicDataDict.CELL_KEY,):
            if key in output:
                # flatten
                val = output[key]
                assert val.shape[-2:] == (3, 3)
                output[key] = val.reshape(val.shape[:-2] + (9,))
        # cartesian tensors like stress are also a special case,
        # we need it to be decomposed into irreps for equivar testing
        for k in cartesian_keys:
            if k in output:
                output[k] = cartesian_tensor[k].from_cartesian(
                    output[k], rtp=cartesian_rtp[k].to(output[k].dtype)
                )
        output_list = []
        # TODO: this is a bit duct-taped and special cased for cell, but maybe it's ok?
        # potentially a more general solution is to inspect the irreps_out to set the dims of the dummy tensors accordingly
        # but the irreps_out values are strings (sometimes?)
        for k in irreps_out:
            if k in output:
                output_list.append(output[k])
            elif k == AtomicDataDict.CELL_KEY:
                # add a dummy cell
                output_list.append(torch.zeros(9, dtype=dtype, device=device))
            else:
                output_list.append(torch.zeros((1,), dtype=dtype, device=device))
        return output_list

    # prepare input data
    for d in data_in:
        # cell is a special case
        if AtomicDataDict.CELL_KEY in d:
            # flatten
            cell = d[AtomicDataDict.CELL_KEY]
            assert cell.shape[-2:] == (3, 3)
            d[AtomicDataDict.CELL_KEY] = cell.reshape(cell.shape[:-2] + (9,))

    errs = [
        equivariance_error(
            wrapper,
            args_in=[d[k] for k in irreps_in],
            irreps_in=list(irreps_in.values()),
            irreps_out=list(irreps_out.values()),
            **kwargs,
        )
        for d in data_in
    ]

    # take max across errors
    errs = {k: torch.max(torch.vstack([e[k] for e in errs]), dim=0)[0] for k in errs[0]}

    current_dtype = (
        func.model_dtype if isinstance(func, GraphModel) else torch.get_default_dtype()
    )
    if e3_tolerance is None:
        e3_tolerance = FLOAT_TOLERANCE[current_dtype]
    all_errs = []
    for case, err in errs.items():
        for key, this_err in zip(irreps_out.keys(), err):
            all_errs.append(case + (key, this_err))
    is_problem = [e[-1] > e3_tolerance for e in all_errs]

    message = (permutation_message + "\n") + "\n".join(
        f"   (parity_k={int(k[0]):1d}, did_translate={str(bool(k[1])):5}, field={str(k[2]):22})     -> max error={float(k[3]):.3e}{'  FAIL' if prob else ''}"
        for k, prob in zip(all_errs, is_problem)
        if irreps_out[str(k[2])] is not None
    )

    if any(is_problem) or " FAIL" in permutation_message:
        raise AssertionError(
            f"Equivariance test of {type(func).__name__} failed:\n   default dtype: {torch.get_default_dtype()} (assumed) model dtype: {current_dtype}  E(3) tolerance: {e3_tolerance}\n{message}"
        )

    return message


_DEBUG_HOOKS = None


def set_irreps_debug(enabled: bool = False) -> None:
    r"""Add debugging hooks to ``forward()`` that check data-irreps consistancy.

    Args:
        enabled: whether to set debug mode as enabled or disabled
    """
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
        if not (isinstance(inp, dict)):
            raise TypeError(
                f"Module {mname} should have received a dict, instead got a {type(inp).__name__}"
            )
        for k, ir in mod.irreps_in.items():
            if k not in inp:
                pass
            elif isinstance(inp[k], torch.Tensor) and isinstance(ir, o3.Irreps):
                if inp[k].ndim == 1 and inp[k].numel() > 0:
                    raise ValueError(
                        f"Field {k} in input to module {mname} has only one dimension (assumed to be batch-like); it must have a second irreps dimension even if irreps.dim == 1 (i.e. a single per atom scalar must have shape [N_at, 1], not [N_at])"
                    )
                elif inp[k].shape[-1] != ir.dim and inp[k].numel() > 0:
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
        if not (isinstance(out, dict)):
            raise TypeError(
                f"Module {mname} should have returned a dict, instead got a {type(out).__name__}"
            )
        for k, ir in mod.irreps_out.items():
            if k not in out:
                pass
            elif isinstance(out[k], torch.Tensor) and isinstance(ir, o3.Irreps):
                if out[k].ndim == 1 and out[k].numel() > 0:
                    raise ValueError(
                        f"Field {k} in output from module {mname} has only one dimension (assumed to be batch-like); it must have a second irreps dimension even if irreps.dim == 1 (i.e. a single per atom scalar must have shape [N_at, 1], not [N_at])"
                    )
                elif out[k].shape[-1] != ir.dim and out[k].numel() > 0:
                    raise ValueError(
                        f"Field {k} in output from {mname} has last dimension {out[k].shape[-1]} but its irreps {ir} indicate last dimension {ir.dim}"
                    )
        return

    h2 = torch.nn.modules.module.register_module_forward_hook(post_hook)

    _DEBUG_HOOKS = (h1, h2)
    return


def override_irreps_debug(enabled=True):
    """Decorator that toggles `set_irreps_debug` at the start of the test
    and restores it to the original state at the end.

    This decorator is crucial for PT2 compilation tests. (the hook modifications in `set_irreps_debug` may interfere with the compilation process).

    Args:
        enabled (bool): whether to enable or disable irreps debug mode
    """

    def decorator(test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            original_state = _DEBUG_HOOKS is not None
            set_irreps_debug(enabled)
            try:
                return test_func(*args, **kwargs)
            finally:
                set_irreps_debug(original_state)

        return wrapper

    return decorator


def edgeset_from_AtomicDataDict(data, **nl_kwargs):
    data = compute_neighborlist_(data, **nl_kwargs)
    return set([tuple(edge) for edge in data["edge_index"].numpy().T])


def compare_neighborlists(
    atoms_or_data: Union[ase.Atoms, AtomicDataDict.Type],
    nl1: str,
    nl2: str,
    **nl_kwargs,
):
    """
    Args:
        nl1, nl2: the neighborlists to compare -- currently "ase", "matscipy", "vesin"
    """
    assert "r_max" in nl_kwargs
    assert "NL" not in nl_kwargs
    if isinstance(atoms_or_data, ase.Atoms):
        data = from_ase(atoms_or_data)
    else:
        data = atoms_or_data
    edges1 = edgeset_from_AtomicDataDict(data, NL=nl1, **nl_kwargs)
    edges2 = edgeset_from_AtomicDataDict(data, NL=nl1, **nl_kwargs)
    assert edges1 == edges2
