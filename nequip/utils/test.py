from typing import Union, Optional, List

import torch
from e3nn import o3
from e3nn.util.test import equivariance_error, FLOAT_TOLERANCE

from nequip.nn import GraphModuleMixin
from nequip.data import (
    AtomicData,
    AtomicDataDict,
    _NODE_FIELDS,
    _EDGE_FIELDS,
)


PERMUTATION_FLOAT_TOLERANCE = {torch.float32: 1e-5, torch.float64: 1e-10}


# https://discuss.pytorch.org/t/how-to-quickly-inverse-a-permutation-by-using-pytorch/116205/4
def _inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


def assert_permutation_equivariant(
    func: GraphModuleMixin,
    data_in: AtomicDataDict.Type,
    tolerance: Optional[float] = None,
):
    r"""Test the permutation equivariance of ``func``.

    Standard fields are assumed to be equivariant to node or edge permutations according to their standard interpretions; all other fields are assumed to be invariant to all permutations. Non-standard fields can be registered as node/edge permutation equivariant using ``register_fields``.

    Raises ``AssertionError`` if issues are found.

    Args:
        func: the module or model to test
        data_in: the example input data to test with
    """
    # Prevent pytest from showing this function in the traceback
    # __tracebackhide__ = True

    if tolerance is None:
        atol = PERMUTATION_FLOAT_TOLERANCE[torch.get_default_dtype()]
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
        if not torch.all(node_perm == torch.arange(n_node, device=device)):
            break
    n_edge: int = data_in[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
    while True:
        edge_perm = torch.randperm(n_edge, device=device)
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

    problems = []
    for k in out_orig.keys():
        if k in node_permute_fields:
            if not torch.allclose(out_orig[k][node_perm], out_perm[k], atol=atol):
                err = (out_orig[k][node_perm] - out_perm[k]).abs().max()
                problems.append(
                    f"node permutation equivariance violated for field {k}; maximum componentwise error: {err:e}"
                )
        elif k in edge_permute_fields:
            if not torch.allclose(out_orig[k][edge_perm], out_perm[k], atol=atol):
                err = (out_orig[k][edge_perm] - out_perm[k]).abs().max()
                problems.append(
                    f"edge permutation equivariance violated for field {k}; maximum componentwise error: {err:e}"
                )
        elif k == AtomicDataDict.EDGE_INDEX_KEY:
            pass
        else:
            # Assume invariant
            if out_orig[k].dtype == torch.bool:
                if not torch.all(out_orig[k] == out_perm[k]):
                    problems.append(
                        f"edge/node permutation invariance violated for field {k} ({k} was assumed to be invariant, should it have been marked as equivariant?)"
                    )
            else:
                if not torch.allclose(out_orig[k], out_perm[k], atol=atol):
                    err = (out_orig[k] - out_perm[k]).abs().max()
                    problems.append(
                        f"edge/node permutation invariance violated for field {k}; maximum componentwise error: {err:e}. (`{k}` was assumed to be invariant, should it have been marked as equivariant?)"
                    )
    if len(problems) > 0:
        raise AssertionError("\n".join(problems))
    return


def assert_AtomicData_equivariant(
    func: GraphModuleMixin,
    data_in: Union[
        AtomicData, AtomicDataDict.Type, List[Union[AtomicData, AtomicDataDict.Type]]
    ],
    permutation_tolerance: Optional[float] = None,
    o3_tolerance: Optional[float] = None,
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
    data_in = [AtomicData.to_AtomicDataDict(d) for d in data_in]

    # == Test permutation of graph nodes ==
    # TODO: since permutation is distinct, run only on one.
    assert_permutation_equivariant(func, data_in[0], tolerance=permutation_tolerance)

    # == Test rotation, parity, and translation using e3nn ==
    irreps_in = {k: None for k in AtomicDataDict.ALLOWED_KEYS}
    irreps_in.update(func.irreps_in)
    irreps_in = {k: v for k, v in irreps_in.items() if k in data_in[0]}
    irreps_out = func.irreps_out.copy()
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
                "3x1o"
            )
            # must be this to actually rotate it
            irps[AtomicDataDict.CELL_KEY] = "3x1o"

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
        return [output[k] for k in irreps_out]

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

    if o3_tolerance is None:
        o3_tolerance = FLOAT_TOLERANCE[torch.get_default_dtype()]
    anerr = next(iter(errs.values()))
    if isinstance(anerr, float) or anerr.ndim == 0:
        # old e3nn doesn't report which key
        problems = {k: v for k, v in errs.items() if v > o3_tolerance}

        def _describe(errors):
            return "\n".join(
                "(parity_k={:d}, did_translate={}) -> max error={:.3e}".format(
                    int(k[0]),
                    bool(k[1]),
                    float(v),
                )
                for k, v in errors.items()
            )

        if len(problems) > 0:
            raise AssertionError(
                "Equivariance test failed for cases:" + _describe(problems)
            )

        return _describe(errs)
    else:
        # it's newer and tells us which is which
        all_errs = []
        for case, err in errs.items():
            for key, this_err in zip(irreps_out.keys(), err):
                all_errs.append(case + (key, this_err))
        problems = [e for e in all_errs if e[-1] > o3_tolerance]

        def _describe(errors):
            return "\n".join(
                "   (parity_k={:1d}, did_translate={:5}, field={:20}) -> max error={:.3e}".format(
                    int(k[0]), str(bool(k[1])), str(k[2]), float(k[3])
                )
                for k in errors
            )

        if len(problems) > 0:
            raise AssertionError(
                "Equivariance test failed for cases:\n" + _describe(problems)
            )

        return _describe(all_errs)


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
    from nequip.utils.torch_geometric import Data

    def pre_hook(mod: GraphModuleMixin, inp):
        # __tracebackhide__ = True
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
        # __tracebackhide__ = True
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
