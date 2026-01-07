# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from nequip.data import AtomicDataDict, compute_neighborlist_
from nequip.data._key_registry import get_field_type
from typing import Optional, Dict, Union, List


class NeighborListTransform(torch.nn.Module):
    """Constructs a neighborlist and adds it to the ``AtomicDataDict``.

    Args:
        r_max (float): cutoff radius used for nearest neighbors
        per_edge_type_cutoff (Dict): optional per-edge-type cutoffs (must be <= r_max)
        type_names (List[str]): list of atom type names
    """

    def __init__(
        self,
        r_max: float,
        per_edge_type_cutoff: Optional[
            Dict[str, Union[float, Dict[str, float]]]
        ] = None,
        type_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()

        self.r_max = r_max
        self.type_names = type_names
        self.per_edge_type_cutoff = per_edge_type_cutoff
        self.kwargs = kwargs

        # set up pruning transform for per-edge-type cutoffs if provided
        self._pruner = None
        if per_edge_type_cutoff is not None:
            assert type_names is not None, (
                "`type_names` required for `per_edge_type_cutoff`"
            )
            self._pruner = NeighborListPruneTransform(
                r_max=self.r_max,
                per_edge_type_cutoff=per_edge_type_cutoff,
                type_names=type_names,
            )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = compute_neighborlist_(data, self.r_max, **self.kwargs)

        # prune based on per-edge-type cutoffs if specified
        if self._pruner is not None:
            data = self._pruner(data)

        return data


class NeighborListPruneTransform(torch.nn.Module):
    """Prunes a neighborlist based on per-edge-type cutoffs.

    Args:
        r_max (float): global cutoff radius
        per_edge_type_cutoff (Dict): per-edge-type cutoffs (must be <= r_max)
        type_names (List[str]): list of atom type names
    """

    def __init__(
        self,
        r_max: float,
        per_edge_type_cutoff: Dict[str, Union[float, Dict[str, float]]],
        type_names: List[str],
    ):
        super().__init__()

        self.r_max = r_max
        self.per_edge_type_cutoff = per_edge_type_cutoff
        self.type_names = type_names

        # set up normalizer for per-edge-type cutoffs
        from nequip.nn.embedding import EdgeLengthNormalizer

        self._normalizer = EdgeLengthNormalizer(
            r_max=self.r_max,
            type_names=type_names,
            per_edge_type_cutoff=per_edge_type_cutoff,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """Prune neighbor list based on per-edge-type cutoffs."""

        if AtomicDataDict.ATOM_TYPE_KEY not in data:
            raise KeyError(
                f"Per-edge-type cutoffs require '{AtomicDataDict.ATOM_TYPE_KEY}' to be present in the data. "
                "This is likely because nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper needs to be applied before "
                "neighborlist pruning. Please check your data transform order."
            )

        # get mask for pruning
        # make a shallow copy of original data dict, such that the original
        # doesn't have the tensors added by applying `self._normalizer`
        mask = (
            self._normalizer(data.copy())[AtomicDataDict.NORM_LENGTH_KEY].view(-1)
            <= 1.0
        )

        # mask edge index (handled separately since it has shape [2, num_edges])
        data[AtomicDataDict.EDGE_INDEX_KEY] = data[AtomicDataDict.EDGE_INDEX_KEY][
            :, mask
        ]

        # mask all other edge fields
        for field in list(data.keys()):
            if field == AtomicDataDict.EDGE_INDEX_KEY:
                continue  # already handled above
            if get_field_type(field, error_on_unregistered=False) == "edge":
                data[field] = data[field][mask]

        return data


class SortedNeighborListTransform(NeighborListTransform):
    """Behaves like :class:`NeighborListTransform` but additionally sorts the neighborlist and provides transpose permutation indices."""

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # first compute the basic neighborlist
        data = super().forward(data)

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
