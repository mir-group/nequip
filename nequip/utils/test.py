import torch
from e3nn.util.test import assert_equivariant

from nequip.data import AtomicData, AtomicDataDict


# TODO update or remove this
def assert_AtomicData_equivariant(
    func, data_in, func_irreps_out, out_field, randomize_features=False, **kwargs
):
    # Prevent pytest from showing this function in the traceback
    __tracebackhide__ = True

    nspecies = len(torch.unique(data_in[AtomicDataDict.ATOMIC_NUMBERS_KEY]))

    key_order = [
        AtomicDataDict.POSITIONS_KEY,
        AtomicDataDict.EDGE_INDEX_KEY,
        AtomicDataDict.EDGE_CELL_SHIFT_KEY,
        AtomicDataDict.CELL_KEY,
        AtomicDataDict.NODE_FEATURES_KEY,
        AtomicDataDict.NODE_ATTRS_KEY,
        AtomicDataDict.BATCH_KEY,
        AtomicDataDict.ATOMIC_NUMBERS_KEY,
    ]
    irreps_in = [
        "3x0e",
        None,
        None,
        None,
        None,  # f"{nspecies}x0e",
        None,  # f"{nspecies}x0e",
        None,
        None,
    ]
    irreps_out = [func_irreps_out]

    def wrapper(*args):
        d = AtomicData(**{k: v for k, v in zip(key_order, args) if v is not None})
        output = func(AtomicData.to_AtomicDataDict(d))
        return output[out_field]

    args_in = [data_in[k] if k in data_in else None for k in key_order]

    if randomize_features:
        for key in [AtomicDataDict.NODE_FEATURES_KEY, AtomicDataDict.NODE_ATTRS_KEY]:
            if key in func.irreps_in:
                args_in[key_order.index(key)] = func.irreps_in[key].randn(
                    data_in.num_nodes, -1
                )

    assert_equivariant(
        wrapper, args_in=args_in, irreps_in=irreps_in, irreps_out=irreps_out, **kwargs
    )
