# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import warnings
from typing import Optional, Union, Dict, List


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
