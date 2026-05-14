# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
import ase.data

from nequip.data import AtomicDataDict

from typing import List, Dict, Optional


class ChemicalSpeciesToAtomTypeMapper(torch.nn.Module):
    """Maps atomic numbers to atom types and adds the atom types to the ``AtomicDataDict``.

    The model operates on abstract atom type indices rather than chemical species, so this transform bridges the gap by mapping atomic numbers to the model's type indices. In the common case, model type names correspond directly to chemical symbols (e.g. ``["H", "C", "O"]``) and the mapping is an identity. Custom type names (e.g. ``"my_Cu"`` instead of ``"Cu"``) require an explicit map.

    Args:
        model_type_names (List[str]): list of atom type names known by the model, e.g. ``["H", "C", "O"]``
        chemical_species_to_atom_type_map (Dict[str, str]): mapping from chemical species to model atom type names, e.g. ``{"H": "H", "C": "C", "O": "O"}`` or ``{"Cu": "my_Cu"}`` for custom type names. Not all ``model_type_names`` need to be present in the map (useful for models trained on full periodic table but simulating subset of elements). If ``None``, defaults to identity mapping, which requires that ``model_type_names`` correspond exactly to chemical species (e.g. ``["H", "C", "O"]``).
    """

    def __init__(
        self,
        model_type_names: Optional[List[str]] = None,
        chemical_species_to_atom_type_map: Optional[Dict[str, str]] = None,
        chemical_symbols: Optional[List[str]] = None,
    ):
        super().__init__()

        # TODO: eventually remove all this logic
        # error out with deprecated API usage
        if chemical_symbols is not None:
            raise ValueError(
                "The `chemical_symbols` parameter is no longer supported. "
                "Please update your config to use `model_type_names` and `chemical_species_to_atom_type_map` instead.\n\n"
                "Old format:\n"
                "  - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper\n"
                "    chemical_symbols: [H, C, O]\n\n"
                "New format:\n"
                "  - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper\n"
                "    model_type_names: [H, C, O]\n"
                "    chemical_species_to_atom_type_map: {H: H, C: C, O: O}\n\n"
                "Or using the list_to_identity_dict resolver:\n"
                "  - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper\n"
                "    model_type_names: [H, C, O]\n"
                "    chemical_species_to_atom_type_map: ${list_to_identity_dict:[H, C, O]}\n"
            )

        # necessary for catching deprecated argument so that we can explain to users the new convention
        if model_type_names is None:
            raise ValueError(
                "ChemicalSpeciesToAtomTypeMapper requires 'model_type_names' argument"
            )

        # default to identity mapping if not provided
        if chemical_species_to_atom_type_map is None:
            chemical_species_to_atom_type_map = {x: x for x in model_type_names}

        # store model_type_names for reference
        self.model_type_names = model_type_names

        # build type_name -> index mapping
        type_name_to_index = {
            name: idx for idx, name in enumerate(self.model_type_names)
        }

        # validate that all mapped atom types exist in model_type_names
        for chem_symbol, atom_type_name in chemical_species_to_atom_type_map.items():
            if atom_type_name not in type_name_to_index:
                raise ValueError(
                    f"Chemical species '{chem_symbol}' maps to atom type '{atom_type_name}', but '{atom_type_name}' is not in model_type_names {self.model_type_names}"
                )

        # make a lookup table mapping atomic numbers to 0-based model type indexes
        lookup_table = torch.full(
            (max(ase.data.atomic_numbers.values()),), -1, dtype=torch.long
        )

        # populate lookup table: chemical symbol -> atomic number -> model type index
        for chem_symbol, atom_type_name in chemical_species_to_atom_type_map.items():
            if chem_symbol not in ase.data.atomic_numbers:
                raise ValueError(f"Invalid chemical symbol '{chem_symbol}'")
            atomic_num = ase.data.atomic_numbers[chem_symbol]
            type_idx = type_name_to_index[atom_type_name]
            lookup_table[atomic_num] = type_idx

        self.register_buffer("lookup_table", lookup_table)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if AtomicDataDict.ATOM_TYPE_KEY in data:
            raise RuntimeError(f"Data already contains {AtomicDataDict.ATOM_TYPE_KEY}")

        atomic_numbers = data[AtomicDataDict.ATOMIC_NUMBERS_KEY]
        data[AtomicDataDict.ATOM_TYPE_KEY] = torch.index_select(
            self.lookup_table, 0, atomic_numbers
        )

        if data[AtomicDataDict.ATOM_TYPE_KEY].min() < 0:
            raise KeyError(
                "Got an atomic number as input for a chemical species that was not specified in `chemical_species_to_atom_type_map`"
            )
        return data
