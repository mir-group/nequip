from typing import Union

import torch
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

    assert_equivariant(
        wrapper,
        args_in=args_in,
        irreps_in=list(irreps_in.values()),
        irreps_out=list(func.irreps_out.values()),
        **kwargs
    )
