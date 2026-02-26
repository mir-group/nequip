# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import warnings
from typing import Optional, Union, Dict, List, Callable

from nequip.data._nl import DEFAULT_NEIGHBORLIST_BACKEND


def handle_chemical_species_map(
    chemical_species_to_atom_type_map: Optional[Union[Dict[str, str], bool]],
    type_names: List[str],
) -> Dict[str, str]:
    """Handle chemical species map fallback to identity map with warning."""
    if chemical_species_to_atom_type_map is None:
        warnings.warn(
            "Defaulting to using model type names as chemical symbols. "
            "If the model type names correspond exactly to chemical species (e.g., 'H', 'C', 'O'), this is correct. "
            "Otherwise, this is wrong and will cause errors. "
            "To silence this warning, explicitly set `chemical_species_to_atom_type_map=True` for identity mapping, "
            "or provide the correct mapping as a dict."
        )
        chemical_species_to_atom_type_map = {t: t for t in type_names}
    elif chemical_species_to_atom_type_map is True:
        # explicitly requested identity mapping without warning
        chemical_species_to_atom_type_map = {t: t for t in type_names}
    return chemical_species_to_atom_type_map


def basic_transforms(
    metadata: dict,
    r_max: float,
    type_names: List[str],
    chemical_species_to_atom_type_map: Dict[str, str],
    neighborlist_backend: str = DEFAULT_NEIGHBORLIST_BACKEND,
) -> List[Callable]:
    """Create transform list with neighborlist construction and optional per-edge-type cutoff pruning."""
    from nequip.data.transforms import (
        ChemicalSpeciesToAtomTypeMapper,
        NeighborListTransform,
    )
    from nequip.nn import graph_model
    from nequip.nn.embedding.utils import cutoff_str_to_fulldict

    transforms = [
        ChemicalSpeciesToAtomTypeMapper(
            model_type_names=type_names,
            chemical_species_to_atom_type_map=chemical_species_to_atom_type_map,
        )
    ]

    if metadata.get(graph_model.PER_EDGE_TYPE_CUTOFF_KEY, None) is not None:
        per_edge_type_cutoff = metadata[graph_model.PER_EDGE_TYPE_CUTOFF_KEY]
        if isinstance(per_edge_type_cutoff, str):
            per_edge_type_cutoff = cutoff_str_to_fulldict(
                per_edge_type_cutoff, type_names
            )

        transforms.append(
            NeighborListTransform(
                r_max=r_max,
                per_edge_type_cutoff=per_edge_type_cutoff,
                type_names=type_names,
                backend=neighborlist_backend,
            )
        )
    else:
        transforms.append(
            NeighborListTransform(r_max=r_max, backend=neighborlist_backend)
        )

    return transforms
