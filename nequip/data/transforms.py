from typing import Dict, Optional, Union, List
import warnings

import torch

import ase.data

from nequip.data import AtomicData, AtomicDataDict


class TypeMapper:
    """Based on a configuration, map atomic numbers to types."""

    num_types: int
    chemical_symbol_to_type: Optional[Dict[str, int]]
    type_to_chemical_symbol: Optional[Dict[int, str]]
    type_names: List[str]
    _min_Z: int

    def __init__(
        self,
        type_names: Optional[List[str]] = None,
        chemical_symbol_to_type: Optional[Dict[str, int]] = None,
        type_to_chemical_symbol: Optional[Dict[int, str]] = None,
        chemical_symbols: Optional[List[str]] = None,
    ):
        if chemical_symbols is not None:
            if chemical_symbol_to_type is not None:
                raise ValueError(
                    "Cannot provide both `chemical_symbols` and `chemical_symbol_to_type`"
                )
            # repro old, sane NequIP behaviour
            # checks also for validity of keys
            atomic_nums = [ase.data.atomic_numbers[sym] for sym in chemical_symbols]
            # https://stackoverflow.com/questions/29876580/how-to-sort-a-list-according-to-another-list-python
            chemical_symbols = [
                e[1] for e in sorted(zip(atomic_nums, chemical_symbols))
            ]
            chemical_symbol_to_type = {k: i for i, k in enumerate(chemical_symbols)}
            del chemical_symbols

        if type_to_chemical_symbol is not None:
            type_to_chemical_symbol = {
                int(k): v for k, v in type_to_chemical_symbol.items()
            }
            assert all(
                v in ase.data.chemical_symbols for v in type_to_chemical_symbol.values()
            )

        # Build from chem->type mapping, if provided
        self.chemical_symbol_to_type = chemical_symbol_to_type
        if self.chemical_symbol_to_type is not None:
            # Validate
            for sym, type in self.chemical_symbol_to_type.items():
                assert sym in ase.data.atomic_numbers, f"Invalid chemical symbol {sym}"
                assert 0 <= type, f"Invalid type number {type}"
            assert set(self.chemical_symbol_to_type.values()) == set(
                range(len(self.chemical_symbol_to_type))
            )
            if type_names is None:
                # Make type_names
                type_names = [None] * len(self.chemical_symbol_to_type)
                for sym, type in self.chemical_symbol_to_type.items():
                    type_names[type] = sym
            else:
                # Make sure they agree on types
                # We already checked that chem->type is contiguous,
                # so enough to check length since type_names is a list
                assert len(type_names) == len(self.chemical_symbol_to_type)
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
            self._index_to_Z = torch.zeros(
                size=(len(self.chemical_symbol_to_type),), dtype=torch.long
            )
            for sym, type_idx in self.chemical_symbol_to_type.items():
                self._index_to_Z[type_idx] = ase.data.atomic_numbers[sym]
            self._valid_set = set(valid_atomic_numbers)
            true_type_to_chemical_symbol = {
                type_id: sym for sym, type_id in self.chemical_symbol_to_type.items()
            }
            if type_to_chemical_symbol is not None:
                assert type_to_chemical_symbol == true_type_to_chemical_symbol
            else:
                type_to_chemical_symbol = true_type_to_chemical_symbol

        # check
        if type_names is None:
            raise ValueError(
                "None of chemical_symbols, chemical_symbol_to_type, nor type_names was provided; exactly one is required"
            )
        # validate type names
        assert all(
            n.isalnum() for n in type_names
        ), "Type names must contain only alphanumeric characters"
        # Set to however many maps specified -- we already checked contiguous
        self.num_types = len(type_names)
        # Check type_names
        self.type_names = type_names
        self.type_to_chemical_symbol = type_to_chemical_symbol
        if self.type_to_chemical_symbol is not None:
            assert set(type_to_chemical_symbol.keys()) == set(range(self.num_types))

    def __call__(
        self, data: Union[AtomicDataDict.Type, AtomicData], types_required: bool = True
    ) -> Union[AtomicDataDict.Type, AtomicData]:
        if AtomicDataDict.ATOM_TYPE_KEY in data:
            if AtomicDataDict.ATOMIC_NUMBERS_KEY in data:
                warnings.warn(
                    "Data contained both ATOM_TYPE_KEY and ATOMIC_NUMBERS_KEY; ignoring ATOMIC_NUMBERS_KEY"
                )
        elif AtomicDataDict.ATOMIC_NUMBERS_KEY in data:
            assert (
                self.chemical_symbol_to_type is not None
            ), "Atomic numbers provided but there is no chemical_symbols/chemical_symbol_to_type mapping!"
            atomic_numbers = data[AtomicDataDict.ATOMIC_NUMBERS_KEY]
            del data[AtomicDataDict.ATOMIC_NUMBERS_KEY]

            data[AtomicDataDict.ATOM_TYPE_KEY] = self.transform(atomic_numbers)
        else:
            if types_required:
                raise KeyError(
                    "Data doesn't contain any atom type information (ATOM_TYPE_KEY or ATOMIC_NUMBERS_KEY)"
                )
        return data

    def transform(self, atomic_numbers):
        """core function to transform an array to specie index list"""

        if atomic_numbers.min() < self._min_Z or atomic_numbers.max() > self._max_Z:
            bad_set = set(torch.unique(atomic_numbers).cpu().tolist()) - self._valid_set
            raise ValueError(
                f"Data included atomic numbers {bad_set} that are not part of the atomic number -> type mapping!"
            )

        return self._Z_to_index.to(device=atomic_numbers.device)[
            atomic_numbers - self._min_Z
        ]

    def untransform(self, atom_types):
        """Transform atom types back into atomic numbers"""
        return self._index_to_Z[atom_types].to(device=atom_types.device)

    @property
    def has_chemical_symbols(self) -> bool:
        return self.chemical_symbol_to_type is not None

    @staticmethod
    def format(
        data: list, type_names: List[str], element_formatter: str = ".6f"
    ) -> str:
        data = torch.as_tensor(data) if data is not None else None
        if data is None:
            return f"[{', '.join(type_names)}: None]"
        elif data.ndim == 0:
            return (f"[{', '.join(type_names)}: {{:{element_formatter}}}]").format(data)
        elif data.ndim == 1 and len(data) == len(type_names):
            return (
                "["
                + ", ".join(
                    f"{{{i}[0]}}: {{{i}[1]:{element_formatter}}}"
                    for i in range(len(data))
                )
                + "]"
            ).format(*zip(type_names, data))
        else:
            raise ValueError(
                f"Don't know how to format data=`{data}` for types {type_names} with element_formatter=`{element_formatter}`"
            )
