# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from nequip.data import AtomicDataDict, compute_neighborlist_


class NeighborListTransform:
    """Constructs a neighborlist and adds it to the ``AtomicDataDict``.

    Args:
        r_max (float): cutoff radius used for nearest neighbors
    """

    def __init__(
        self,
        r_max: float,
        **kwargs,
    ):
        self.r_max = r_max
        self.kwargs = kwargs

    def __call__(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = compute_neighborlist_(data, self.r_max, **self.kwargs)
        return data


class SortedNeighborListTransform(NeighborListTransform):
    """Constructs a sorted neighborlist with permutation indices for a sorted transpose.

    Args:
        r_max (float): cutoff radius used for nearest neighbors
    """

    def __call__(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # first compute the basic neighborlist
        data = super().__call__(data)

        # sort the edge index and corresponding edge attributes
        edge_idxs = data[AtomicDataDict.EDGE_INDEX_KEY]

        # short-circuit for empty edge index case
        if edge_idxs.numel() == 0:
            return data

        sort_indices = torch.argsort(
            edge_idxs[0] * AtomicDataDict.num_nodes(data) + edge_idxs[1]
        )

        # sort edge indices
        data[AtomicDataDict.EDGE_INDEX_KEY] = torch.index_select(
            edge_idxs, 1, sort_indices
        )
        edge_idxs = data[AtomicDataDict.EDGE_INDEX_KEY]

        # sort cell shifts if present
        if AtomicDataDict.EDGE_CELL_SHIFT_KEY in data:
            data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = torch.index_select(
                data[AtomicDataDict.EDGE_CELL_SHIFT_KEY], 0, sort_indices
            )

        # compute transpose permutation for backward pass
        # compute full transpose permutation: sort by sender first, then receiver
        transpose_perm = torch.argsort(
            edge_idxs[1] * AtomicDataDict.num_nodes(data) + edge_idxs[0]
        )
        data[AtomicDataDict.EDGE_TRANSPOSE_PERM_KEY] = transpose_perm

        return data
