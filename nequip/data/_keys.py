# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""Keys for dictionaries/AtomicData objects.

This is a seperate module to compensate for a TorchScript bug that can only recognize constants when they are accessed as attributes of an imported module.
"""

from typing import List, Final


# Define allowed keys as constants

# === core ===
# (num_atoms, 3) | positions of the atoms in the system
POSITIONS_KEY: Final[str] = "pos"
# (num_atoms, 1) | long tensor of atom types (not necessarily corresponding to elements)
ATOM_TYPE_KEY: Final[str] = "atom_types"
# (2, num_edges) | long tensor giving center -> neighbor relations
# [0] entry corresponds to desination (dst) nodes
# [1] entry corresponds to source (src) nodes
EDGE_INDEX_KEY: Final[str] = "edge_index"

# permutation indices for transposing edges from row to column major order
EDGE_TRANSPOSE_PERM_KEY: Final[str] = "edge_transpose_perm"

# === cell related ===
# (num_frames, 3, 3) | rows are the cell vectors
CELL_KEY: Final[str] = "cell"
# (num_edges, 3) | how many periodic cells each edge crosses in each cell vector
EDGE_CELL_SHIFT_KEY: Final[str] = "edge_cell_shift"

# === batching keys ===
# used in the context of training and batched inference (e.g. torchsim)
# (num_atoms,) | long tensor mapping each atom to its frame/graph index
BATCH_KEY: Final[str] = "batch"
# (num_frames,) | long tensor giving the number of atoms in each frame/graph
NUM_NODES_KEY: Final[str] = "num_atoms"

# === usually unused by model, but present in data ===
# (num_frames, 3) | bool tensor corresponding to periodic boundary conditions
PBC_KEY: Final[str] = "pbc"
# (num_atoms, 1) | long tensor of atomic numbers of elements
ATOMIC_NUMBERS_KEY: Final[str] = "atomic_numbers"

# === physical edge quantities ===
# (num_edges, 3) | edge displacement vectors defined by EDGE_INDEX (including periodic image shifts if applicable)
EDGE_VECTORS_KEY: Final[str] = "edge_vectors"
# (num_edges, 1) | Euclidean norm of EDGE_VECTORS, kept as a column for broadcasting
EDGE_LENGTH_KEY: Final[str] = "edge_lengths"
# (num_edges, 1) | EDGE_LENGTH normalized by r_max (or per-edge-type cutoff if used)
NORM_LENGTH_KEY: Final[str] = "normed_edge_lengths"
# (2, num_edges) | long tensor of per-edge atom-type pairs [center(dst), neighbor(src)]
# used to form flattened pair indices as edge_type[0] * num_types + edge_type[1]
# but the tensor is still (2, num_edges), so the flat is historical
# TODO: maybe change the str if we're sure that it won't ripple out with problems
EDGE_TYPE_KEY: Final[str] = "edge_type_flat"

# === feature keys ===
# (num_edges, 1) | invariant radial cutoff envelope per edge, allows reuse of cutoff envelopes
EDGE_CUTOFF_KEY: Final[str] = "edge_cutoff"
# (num_edges, dim) | attributes of each edge (usually used for tensors)
EDGE_ATTRS_KEY: Final[str] = "edge_attrs"
# (num_edges, dim) | embedding of the edges (usually used for scalars)
EDGE_EMBEDDING_KEY: Final[str] = "edge_embedding"
# (num_edges, dim) | learned edge features
EDGE_FEATURES_KEY: Final[str] = "edge_features"

# (num_atoms, dim) | node attributes/conditioning features
NODE_ATTRS_KEY: Final[str] = "node_attrs"
# (num_atoms, dim) | learned node features
NODE_FEATURES_KEY: Final[str] = "node_features"
# (num_atoms, 1) | cached per-node normalization factor for NODE_FEATURES_KEY
FEATURE_NORM_FACTOR_KEY: Final[str] = "feature_norm_factor"

# === base physical predictions ===
# (num_edges, 1) | per-edge energy contributions (used by Allegro)
EDGE_ENERGY_KEY: Final[str] = "edge_energy"
# (num_atoms, 1) | per-atom energy contributions
PER_ATOM_ENERGY_KEY: Final[str] = "atomic_energy"
# (num_frames, 1) | total energy per frame/graph
TOTAL_ENERGY_KEY: Final[str] = "total_energy"
# (num_atoms, 3) | total force on each atom
FORCE_KEY: Final[str] = "forces"
# (num_edges, 3) | dE/d(edge_vectors), used for LAMMPS ML-IAP inference
EDGE_FORCE_KEY: Final[str] = "edge_forces"
# (num_atoms, num_atoms, 3) | pairwise force decomposition; sums over axis 0 to FORCE_KEY
PARTIAL_FORCE_KEY: Final[str] = "partial_forces"
# (num_frames, 3, 3) | stress tensor per frame (symmetric)
STRESS_KEY: Final[str] = "stress"
# (num_frames, 3, 3) | virial tensor per frame (symmetric)
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

TOTAL_CHARGE_KEY: Final[str] = "charge"
TOTAL_SPIN_KEY: Final[str] = "spin"

# metadata for a frame/graph
FRAME_SUBSET_KEY: Final[str] = "subset"

# === LAMMPS MLIAP integration keys ===
LMP_MLIAP_DATA_KEY: Final[str] = "lmp_mliap_data"
# (2,) tensor containing [num_local_atoms, num_ghost_atoms]
# the sum of the entries is `num_total_atoms`
# used with LAMMPS ML-IAP for now
# but possible for reuse with other integrations based on similar local-ghost schemes
NUM_LOCAL_GHOST_NODES_KEY: Final[str] = "num_local_ghost_atoms"

# make a list of allowed keys
ALLOWED_KEYS: List[str] = [v for k, v in globals().items() if k.endswith("_KEY")]
# check that the fields don't have "." (to avoid clashes with nn parameter names)
assert all(["." not in key for key in ALLOWED_KEYS])
