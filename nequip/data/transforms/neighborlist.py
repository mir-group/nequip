from nequip.data import AtomicDataDict


class NeighborListTransform:
    """Constructs or retrieves a neighborlist  for an AtomicDataDict."""

    def __init__(
        self,
        r_max: float,
        **nl_kwargs,
    ):
        self.r_max = r_max
        self.nl_kwargs = nl_kwargs

    def __call__(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        return AtomicDataDict.compute_neighborlist_(data, self.r_max, **self.nl_kwargs)
