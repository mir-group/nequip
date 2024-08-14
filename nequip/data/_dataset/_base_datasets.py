from typing import List, Callable, Union, Optional

import torch
from ..transforms import TypeMapper


class AtomicDataset(torch.utils.data.Dataset):
    """The base class for all NequIP datasets.

    Like any PyTorch dataset, subclasses must implement:
     - `__len__`
     - `__getitem__`
    and optionally but encouraged,
     - `__getitems__`
    """

    root: str
    dtype: torch.dtype
    type_mapper: TypeMapper

    def __init__(
        self,
        root: str,
        type_mapper: Optional[TypeMapper] = None,
    ):
        self.dtype = torch.get_default_dtype()
        self.root = root
        self.type_mapper = type_mapper

    def statistics(
        self,
        fields: List[Union[str, Callable]],
        modes: List[str],
        stride: int = 1,
        unbiased: bool = True,
    ) -> List[tuple]:
        """Compute the statistics of ``fields`` in the dataset.

        If the values at the fields are vectors/multidimensional, they must be of fixed shape
        and elementwise statistics will be computed.

        Args:
            fields: the names of the fields to compute statistics for.
                Instead of a field name, a callable can also be given that reuturns a quantity
                to compute the statisics for.

                If a callable is given, it will be called with a (possibly batched) ``Data``-like
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

            stride: the stride over the dataset while computing statistcs.

            unbiased: whether to use unbiased for standard deviations.

        Returns:
            List of statistics for each field.
        """
        # TODO: pytorch_runstats implementation
        raise NotImplementedError
