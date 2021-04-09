"""Keys for dictionaries/AtomicData objects.

This is a seperate module to compensate for a TorchScript bug that can only recognize constants when they are accessed as attributes of an imported module.
"""
import sys
from typing import List, Final

# == Define allowed keys as constants ==
# The positions of the atoms in the system
POSITIONS_KEY: Final[str] = "pos"
WEIGHTS_KEY: Final[str] = "weights"

# The [2, n_edge] index tensor giving center -> neighbor relations
EDGE_INDEX_KEY: Final[str] = "edge_index"
# A [n_edge, 3] tensor of how many periodic cells each edge crosses in each cell vector
EDGE_CELL_SHIFT_KEY: Final[str] = "edge_cell_shift"
# A [n_edge, 3] tensor of displacement vectors associated to edges
EDGE_VECTORS_KEY: Final[str] = "edge_vectors"
# A [n_edge] tensor of the lengths of EDGE_VECTORS
EDGE_LENGTH_KEY: Final[str] = "edge_lengths"
# [n_edge, dim] (possibly equivariant) attributes of each edge
EDGE_ATTRS_KEY: Final[str] = "edge_attrs"
# [n_edge, dim] invariant embedding of the edges
EDGE_EMBEDDING_KEY: Final[str] = "edge_embedding"

CELL_KEY: Final[str] = "cell"
PBC_KEY: Final[str] = "pbc"

NODE_FEATURES_KEY: Final[str] = "node_features"
NODE_ATTRS_KEY: Final[str] = "node_attrs"
ATOMIC_NUMBERS_KEY: Final[str] = "atomic_numbers"
SPECIES_INDEX_KEY: Final[str] = "species_index"
SPECIES_WEIGHTS_KEY: Final[str] = "species_index"
PER_ATOM_ENERGY_KEY: Final[str] = "atomic_energy"
TOTAL_ENERGY_KEY: Final[str] = "total_energy"
FORCE_KEY: Final[str] = "forces"

BATCH_KEY: Final[str] = "batch"

# Make a list of allowed keys
ALLOWED_KEYS: List[str] = [
    getattr(sys.modules[__name__], k)
    for k in sys.modules[__name__].__dict__.keys()
    if k.endswith("_KEY")
]
