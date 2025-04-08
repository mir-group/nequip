# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
import ase.data

from nequip.data import AtomicDataDict

from typing import List


class ChemicalSpeciesToAtomTypeMapper:
    """Maps atomic numbers to atom types and adds the atom types to the ``AtomicDataDict``.

    This transform accounts for how the atom types seen by the model can be different from the atomic species that one obtains from a conventional dataset. There could be cases where the same chemical species corresponds to multiple atom types, e.g. different charge states.

    The order of the ``chemical_symbols`` list must match the order of the list of ``type_names`` known by the model.

    Args:
        chemical_symbols ([List[str]): list of chemical species
    """

    def __init__(self, chemical_symbols: List[str]):
        # Make a lookup table mapping atomic numbers to 0-based model type indexes
        self.lookup_table = torch.full(
            (max(ase.data.atomic_numbers.values()),), -1, dtype=torch.long
        )
        for idx, sym in enumerate(chemical_symbols):
            assert sym in ase.data.atomic_numbers, f"Invalid chemical symbol {sym}"
            self.lookup_table[ase.data.atomic_numbers[sym]] = idx

    def __call__(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if AtomicDataDict.ATOM_TYPE_KEY in data:
            raise RuntimeError("Data already contains AtomicDataDict.ATOM_TYPE_KEY")

        atomic_numbers = data[AtomicDataDict.ATOMIC_NUMBERS_KEY]
        data[AtomicDataDict.ATOM_TYPE_KEY] = torch.index_select(
            self.lookup_table, 0, atomic_numbers
        )
        if data[AtomicDataDict.ATOM_TYPE_KEY].min() < 0:
            raise KeyError(
                "Got an atomic number as input for a chemical species that was not specified in the `chemical_symbols`"
            )
        return data
