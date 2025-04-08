# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
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
        return compute_neighborlist_(data, self.r_max, **self.kwargs)
