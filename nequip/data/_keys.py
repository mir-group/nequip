"""Keys for dictionaries/AtomicData objects.

This is a seperate module to compensate for a TorchScript bug that can only recognize constants when they are accessed as attributes of an imported module.
"""
import sys
from typing import List

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

# == Define allowed keys as constants ==
# The positions of the atoms in the system
POSITIONS_KEY: Final[str] = "pos"
# The [2, n_edge] index tensor giving center -> neighbor relations
EDGE_INDEX_KEY: Final[str] = "edge_index"
# A [n_edge, 3] tensor of how many periodic cells each edge crosses in each cell vector
EDGE_CELL_SHIFT_KEY: Final[str] = "edge_cell_shift"
# [n_batch, 3, 3] or [3, 3] tensor where rows are the cell vectors
CELL_KEY: Final[str] = "cell"
# [n_batch, 3] bool tensor
PBC_KEY: Final[str] = "pbc"
# [n_atom, 1] long tensor
ATOMIC_NUMBERS_KEY: Final[str] = "atomic_numbers"
# [n_atom, 1] long tensor
ATOM_TYPE_KEY: Final[str] = "atom_types"

BASIC_STRUCTURE_KEYS: Final[List[str]] = [
    POSITIONS_KEY,
    EDGE_INDEX_KEY,
    EDGE_CELL_SHIFT_KEY,
    CELL_KEY,
    PBC_KEY,
    ATOM_TYPE_KEY,
    ATOMIC_NUMBERS_KEY,
]

# A [n_edge, 3] tensor of displacement vectors associated to edges
EDGE_VECTORS_KEY: Final[str] = "edge_vectors"
# A [n_edge] tensor of the lengths of EDGE_VECTORS
EDGE_LENGTH_KEY: Final[str] = "edge_lengths"
# [n_edge, dim] (possibly equivariant) attributes of each edge
EDGE_ATTRS_KEY: Final[str] = "edge_attrs"
# [n_edge, dim] invariant embedding of the edges
EDGE_EMBEDDING_KEY: Final[str] = "edge_embedding"
EDGE_FEATURES_KEY: Final[str] = "edge_features"
# [n_edge, 1] invariant of the radial cutoff envelope for each edge, allows reuse of cutoff envelopes
EDGE_CUTOFF_KEY: Final[str] = "edge_cutoff"
# edge energy as in Allegro
EDGE_ENERGY_KEY: Final[str] = "edge_energy"

NODE_FEATURES_KEY: Final[str] = "node_features"
NODE_ATTRS_KEY: Final[str] = "node_attrs"

PER_ATOM_ENERGY_KEY: Final[str] = "atomic_energy"
TOTAL_ENERGY_KEY: Final[str] = "total_energy"
FORCE_KEY: Final[str] = "forces"
PARTIAL_FORCE_KEY: Final[str] = "partial_forces"
STRESS_KEY: Final[str] = "stress"
VIRIAL_KEY: Final[str] = "virial"

ALL_ENERGY_KEYS: Final[List[str]] = [
    EDGE_ENERGY_KEY,
    PER_ATOM_ENERGY_KEY,
    TOTAL_ENERGY_KEY,
    FORCE_KEY,
    PARTIAL_FORCE_KEY,
    STRESS_KEY,
    VIRIAL_KEY,
]

BATCH_KEY: Final[str] = "batch"
BATCH_PTR_KEY: Final[str] = "ptr"

# Make a list of allowed keys
ALLOWED_KEYS: List[str] = [
    getattr(sys.modules[__name__], k)
    for k in sys.modules[__name__].__dict__.keys()
    if k.endswith("_KEY")
]
