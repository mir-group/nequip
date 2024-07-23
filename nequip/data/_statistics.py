from typing import List, Union, Callable, Iterable

import torch
from torch_runstats.scatter import scatter_mean, scatter_std

from . import AtomicDataDict, _key_registry

import logging

logger = logging.getLogger(__name__)


# TODO: add stride support


def statistics(
    data_source: Iterable[AtomicDataDict.Type],
    fields: List[Union[str, Callable]],
    modes: List[str],
    unbiased: bool = True,
) -> List[tuple]:
    """Compute the statistics of ``fields`` in the dataset.

    If the values at the fields are vectors/multidimensional, they must be of fixed shape
    and elementwise statistics will be computed.

    Args:
        data_source: an iterable of AtomicDataDict.Type. If striding or subsampling of the data
            is necessary to fit in memory or otherwise desired, it should be done before calling
            `statistics` and already be a property of the passed iterable.
        fields: the names of the fields to compute statistics for.
            Instead of a field name, a callable can also be given that returns a quantity
            to compute the statisics for.

            If a callable is given, it will be called with a (possibly batched) `AtomicDataDict.Type`
            object and must return a sequence of points to add to the set over which the statistics
            will be computed. The callable must also return a string, one of ``"node"`` or ``"graph"``,
            indicating whether the value it returns is a per-node or per-graph quantity.

            PLEASE NOTE: the argument to the callable may be "batched", and it may not be batched
            "contiguously": ``batch`` and ``edge_index`` may have "gaps" in their values.

            For example, to compute the overall statistics of the x,y,z components of a per-node vector
            ``force`` field:

                data.statistics([lambda data: (data.force.flatten(), "node")])

            The above computes the statistics over a set of size 3N, where N is the total number of nodes
            in the dataset.

        modes: the statistic to compute for each field. Valid options are:
                - ``count``
                - ``rms``
                - ``mean_std``
                - ``per_atom_*``

        unbiased: whether to use unbiased for standard deviations.

    Returns:
        List of statistics for each field.
    """
    # TODO: pytorch_runstats implementation with truly online statistics?

    # Short circut:
    assert len(modes) == len(fields)
    if len(fields) == 0:
        return []

    # flags to avoid unnecessary operations and memory use
    per_atom: bool = any([mode.startswith("per_atom_") for mode in modes])
    per_type: bool = any([mode.startswith("per_type_") for mode in modes])

    # Go through the data and keep only the necessary requested data
    arrays = [list() for _ in fields]
    type_arrays = [list() for _ in fields]
    if per_atom:
        num_nodes = []
    if per_type:
        atom_types = []

    for data in data_source:
        if per_atom:
            num_nodes.append(data[AtomicDataDict.NUM_NODES_KEY])
        if per_type:
            atom_types.append(data[AtomicDataDict.ATOM_TYPE_KEY])

        # TODO: this seems inefficient?
        for field_idx, field in enumerate(fields):
            if callable(field):
                field_tensor, field_type = field(data)
            else:
                field_tensor = data[field]
                field_type = _key_registry.get_field_type(field)
            arrays[field_idx].append(field_tensor)
            type_arrays[field_idx].append(field_type)
    del data, data_source, field_tensor, field_type, field

    # check if all field_types
    for lst in type_arrays:
        assert _all_same(lst), "field types differ across data entries"
    type_array = [field_type_list[0] for field_type_list in type_arrays]
    del type_arrays

    if per_atom:
        num_nodes_recip = torch.cat(num_nodes, dim=0).double().reciprocal()
    if per_type:
        atom_types = torch.cat(atom_types, dim=0)

    # Now do statistics
    results = []
    for idx, mode in enumerate(modes):
        array = torch.cat(arrays.pop(0), dim=0)
        field = fields.pop(0)
        # TODO: how to have more verbose error messages for callable fields?
        field = f'"callable field at index {idx}"' if callable(field) else field

        field_type = type_array.pop(0)
        assert field_type in ["graph", "node", "edge"]

        if mode.startswith("per_atom_"):
            mode = mode[len("per_atom_") :]
            if field_type != "graph":
                raise ValueError(
                    f"`per_atom_{mode}` only works for per-graph quantities, but found {field_type} quantity {field}."
                )
            array = torch.einsum("b..., b -> b...", array, num_nodes_recip)

        do_per_type = False
        if mode.startswith("per_type_"):
            mode = mode[len("per_type_") :]
            if field_type == "graph":
                raise ValueError(
                    f"`per_type_{mode}` only works for node or edge quantities, but found graph quantity {field}."
                )
            elif field_type == "node":
                do_per_type = True
            elif field_type == "edge":
                raise NotImplementedError(
                    f"Raise a GitHub issue if you want to compute `per_type_{mode}` of edge quantities like {field}"
                )

        if mode == "count":
            if torch.is_floating_point(array):
                raise ValueError(
                    f"Unable to compute {mode} statistics on floating point quantity {field}."
                )
            else:
                if not do_per_type:
                    res = torch.unique(
                        torch.flatten(array), return_counts=True, sorted=True
                    )
                else:
                    raise NotImplementedError("`per_type_count` not yet implemented")
        elif mode == "rms":
            array = array.to(torch.get_default_dtype())
            if not do_per_type:
                res = (torch.sqrt(torch.mean(array.square_(), dim=0)),)
            else:
                array = scatter_mean(array.square_(), atom_types, dim=0)
                array = array.reshape((array.shape[0], -1)).mean(-1)
                res = (torch.sqrt(array),)
        elif mode == "mean_std":
            array = array.to(torch.get_default_dtype())
            if not do_per_type:
                mean = torch.mean(array, dim=0)
                std = torch.std(array, dim=0, unbiased=unbiased)
                res = (mean, std)
            else:
                # TODO: test this feature
                mean = scatter_mean(array, atom_types, dim=0)
                mean = mean.reshape((mean.shape[0], -1)).mean(-1)
                std = scatter_std(array, atom_types, dim=0)
                std = std.reshape((std.shape[0], -1)).square().sum(-1).sqrt()
                res = (mean, std)
        elif mode == "absmax":
            if not do_per_type:
                res = (array.abs_().max(dim=0),)
            else:
                # TODO: need to implement scatter_absmax in torch_runstats
                raise NotImplementedError("`per_type_absmax` not yet implemented")
        else:
            raise NotImplementedError(f"No such statistics mode `{mode}`")

        results.append(res)

    return results


def _all_same(lst):
    return not lst or lst.count(lst[0]) == len(lst)


def compute_stats_for_model(
    field: Union[str, Callable],
    mode: str,
    data_source: Iterable[AtomicDataDict.Type],
):
    """Computes dataset statistics for model building.

    Quantity name should be dataset_key_stat, where key can be any key
    that exists in the dataset, stat can be mean, std

    Args:
        field (Union[str, Callable]): data field or a callable
        mode: ``mean``, ``std``, ``rms``, ``absmax`` with ``per_atom`` or ``per_type`` prefix
        data_source (Iterable[AtomicDataDict.Type]): iterable of AtomicDataDicts
    """
    tuple_id_map = {"mean": 0, "std": 1, "rms": 0, "absmax": 0}

    # identify per_type and per_atom modes
    prefix = ""
    if mode.startswith("per_type_"):
        mode = mode[len("per_type_") :]
        prefix = "per_type_"
    elif mode.startswith("per_atom_"):
        mode = mode[len("per_atom_") :]
        prefix = "per_atom_"

    stat = mode.split("_")[-1]
    if stat in ["mean", "std"]:
        stat_mode = prefix + "mean_std"
    elif stat in ["rms", "absmax"]:
        stat_mode = prefix + stat
    else:
        raise ValueError(f"Cannot handle {stat} type quantity")

    # == special cases and their helper functions ==
    def force_components(data):
        return (data["forces"].flatten(), "node")

    def num_neighbors(data):
        counts = torch.unique(
            data[AtomicDataDict.EDGE_INDEX_KEY][0],
            sorted=True,
            return_counts=True,
        )[1]
        # in case the cutoff is small and some nodes have no neighbors,
        # we need to pad `counts` up to the right length
        counts = torch.nn.functional.pad(
            counts, pad=(0, len(data[AtomicDataDict.POSITIONS_KEY]) - len(counts))
        )
        return (counts, "node")

    if field == "force_components":
        field = force_components
    elif field == "num_neighbors":
        field = num_neighbors

    values = statistics(
        data_source=data_source,
        fields=[field],
        modes=[stat_mode],
    )
    values = values[0][tuple_id_map[stat]]

    field_name = field.__name__ if isinstance(field, Callable) else field
    pretty_values = values.item() if values.numel() == 1 else values
    logger.info(
        f"Computed training dataset statistics {mode} {field_name}: {pretty_values}"
    )

    return values
