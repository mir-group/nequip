# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""Keys for dictionaries/AtomicData objects.

This is a seperate module to compensate for a TorchScript bug that can only recognize constants when they are accessed as attributes of an imported module.
"""

from typing import List, Final


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
NORM_LENGTH_KEY: Final[str] = "normed_edge_lengths"
EDGE_TYPE_KEY: Final[str] = "edge_type_flat"

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

# misc ase property keys
# except for "energy", "energies" (handled with key_mapping)
FREE_ENERGY_KEY: Final[str] = "free_energy"
PER_ATOM_STRESS_KEY: Final[str] = "stresses"
TOTAL_MAGMOM_KEY: Final[str] = "magmom"
MAGMOM_KEY: Final[str] = "magmoms"
CHARGE_KEY: Final[str] = "charges"
DIPOLE_KEY: Final[str] = "dipole"
DIELECTRIC_KEY: Final[str] = "dielectric_tensor"
BORN_CHARGE_KEY: Final[str] = "born_effective_charges"
POLARIZATION_KEY: Final[str] = "polarization"

# metadata for a frame/graph
FRAME_SUBSET_KEY: Final[str] = "subset"

# batch related keys
BATCH_KEY: Final[str] = "batch"
NUM_NODES_KEY: Final[str] = "num_atoms"

# make a list of allowed keys
ALLOWED_KEYS: List[str] = [v for k, v in globals().items() if k.endswith("_KEY")]
# check that the fields don't have "." (to avoid clashes with nn parameter names)
assert all(["." not in key for key in ALLOWED_KEYS])
