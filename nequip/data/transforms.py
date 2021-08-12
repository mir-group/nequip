from typing import Dict, Optional, Union
import warnings

import torch

import ase.data

from nequip.data import AtomicData, AtomicDataDict


class TypeMapper:
    """Based on a configuration, map atomic numbers to types."""

    chemical_symbol_to_type: Optional[Dict[str, int]]
    _min_Z: int

    def __init__(
        self,
        num_species: int,
        chemical_symbol_to_type: Optional[Dict[str, int]] = None,
    ):
        self.chemical_symbol_to_type = chemical_symbol_to_type
        if self.chemical_symbol_to_type is not None:
            # Validate
            for sym, type in self.chemical_symbol_to_type.items():
                assert sym in ase.data.atomic_numbers, f"Invalid chemical symbol {sym}"
                assert 0 <= type < num_species, f"Invalid type number {type}"
            # Make mapper array
            valid_atomic_numbers = [
                ase.data.atomic_numbers[sym] for sym in self.chemical_symbol_to_type
            ]
            self._min_Z = min(valid_atomic_numbers)
            self._max_Z = max(valid_atomic_numbers)
            Z_to_index = torch.full(
                size=(1 + self._max_Z - self._min_Z,), fill_value=-1, dtype=torch.long
            )
            for sym, type in self.chemical_symbol_to_type.items():
                Z_to_index[ase.data.atomic_numbers[sym] - self._min_Z] = type
            self._Z_to_index = Z_to_index

    def __call__(
        self, data: Union[AtomicDataDict.Type, AtomicData]
    ) -> Union[AtomicDataDict.Type, AtomicData]:
        if AtomicDataDict.SPECIES_INDEX_KEY in data:
            if AtomicDataDict.ATOMIC_NUMBERS_KEY in data:
                warnings.warn(
                    "Data contained both SPECIES_INDEX_KEY and ATOMIC_NUMBERS_KEY; ignoring ATOMIC_NUMBERS_KEY"
                )
        elif AtomicDataDict.ATOMIC_NUMBERS_KEY in data:
            assert (
                self.chemical_symbol_to_type is not None
            ), "Atomic numbers provided but there is no chemical_symbol_to_type mapping!"
            atomic_numbers = data[AtomicDataDict.ATOMIC_NUMBERS_KEY]
            # TODO: torch_geometric data doesn't support `del` yet
            delattr(data, AtomicDataDict.ATOMIC_NUMBERS_KEY)
            if atomic_numbers.min() < self._min_Z or atomic_numbers.max() > self._max_Z:
                raise ValueError(
                    "Some provided atomic numbers are not part of the atomic number -> type mapping!"
                )
            data[AtomicDataDict.SPECIES_INDEX_KEY] = self._Z_to_index[
                atomic_numbers - self._min_Z
            ]
            if data[AtomicDataDict.SPECIES_INDEX_KEY].min() < 0:
                raise ValueError(
                    "Some provided atomic numbers are not part of the atomic number -> type mapping!"
                )
        else:
            raise KeyError(
                "Data doesn't contain any atom type information (SPECIES_INDEX_KEY or ATOMIC_NUMBERS_KEY)"
            )
        return data
