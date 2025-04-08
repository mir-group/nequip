# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from copy import deepcopy
import warnings
import numpy as np
import torch

import ase
from ase.calculators.singlepoint import SinglePointCalculator, SinglePointDFTCalculator
from ase.calculators.calculator import all_properties as ase_all_properties
from ase.stress import full_3x3_to_voigt_6_stress

from . import AtomicDataDict, _key_registry
from .dict import from_dict

from typing import Dict, Union, List, Optional


def from_ase(
    atoms: ase.Atoms,
    key_mapping: Optional[Dict[str, str]] = {},
    include_keys: Optional[List] = [],
    exclude_keys: Optional[List] = [],
) -> AtomicDataDict.Type:
    """Build an ``AtomicDataDict`` from an ``ase.Atoms`` object.

    Respects ``atoms``'s ``pbc`` and ``cell``.

    First tries to extract energies and forces from a single-point calculator associated with the ``Atoms`` if one is present and has those fields.
    If either is not found, the method will look for ``energy``/``energies`` and ``force``/``forces`` in ``atoms.arrays``.

    Args:
        atoms (ase.Atoms): the input.
        key_mapping (Optional[Dict]): rename ase property name to a new string name.
        include_keys (Optional[List]): list of additional keys to include in AtomicData aside from the ones defined in ``ase.calculators.calculator.all_properties``
        exclude_keys (Optional[List]): list of keys that may be present in the ``ase.Atoms`` object but the user wishes to exclude
    """
    from nequip.ase import NequIPCalculator

    default_args = set(
        [
            "numbers",
            "positions",
        ]  # ase internal names for position and atomic_numbers
        + [
            AtomicDataDict.PBC_KEY,
            AtomicDataDict.CELL_KEY,
            AtomicDataDict.POSITIONS_KEY,
        ]  # arguments for from_dict method
    )
    include_keys = list(
        set(list(include_keys) + ase_all_properties + list(key_mapping.keys()))
        - default_args
        - set(exclude_keys)
    )

    km = {
        "forces": AtomicDataDict.FORCE_KEY,
        "energy": AtomicDataDict.TOTAL_ENERGY_KEY,
        "energies": AtomicDataDict.PER_ATOM_ENERGY_KEY,
    }
    km.update(key_mapping)
    key_mapping = km

    add_fields = {}

    # Get info from atoms.arrays; lowest priority. copy first
    add_fields = {
        key_mapping.get(k, k): v for k, v in atoms.arrays.items() if k in include_keys
    }

    # Get info from atoms.info; second lowest priority.
    add_fields.update(
        {key_mapping.get(k, k): v for k, v in atoms.info.items() if k in include_keys}
    )

    if atoms.calc is not None:
        if isinstance(atoms.calc, (SinglePointCalculator, SinglePointDFTCalculator)):
            add_fields.update(
                {
                    key_mapping.get(k, k): deepcopy(v)
                    for k, v in atoms.calc.results.items()
                    if k in include_keys
                }
            )
        elif isinstance(atoms.calc, NequIPCalculator):
            pass  # otherwise the calculator breaks
        else:
            raise NotImplementedError(
                f"`from_ase` does not support calculator {atoms.calc}"
            )

    data = {
        AtomicDataDict.POSITIONS_KEY: atoms.positions,
        AtomicDataDict.CELL_KEY: np.array(atoms.get_cell()),
        AtomicDataDict.PBC_KEY: atoms.get_pbc(),
        AtomicDataDict.ATOMIC_NUMBERS_KEY: atoms.get_atomic_numbers(),
    }
    data.update(**add_fields)
    return from_dict(data)


def to_ase(
    data: AtomicDataDict.Type,
    chemical_symbols: Optional[List[str]] = None,
    extra_fields: List[str] = [],
) -> Union[List[ase.Atoms], ase.Atoms]:
    """Build a (list of) ``ase.Atoms`` object(s) from an ``AtomicData`` object.

    For each unique batch number provided in ``AtomicDataDict.BATCH_KEY``,
    an ``ase.Atoms`` object is created. If ``AtomicDataDict.BATCH_KEY`` does not
    exist in self, a single ``ase.Atoms`` object is created.

    Args:
        chemical_symbols: if provided, will be used to map ``ATOM_TYPES`` back into
            elements, if the configuration of the ``type_mapper`` allows.
        extra_fields: fields other than those handled explicitly (currently
            those defining the structure as well as energy, per-atom energy,
            and forces) to include in the output object. Per-atom (per-node)
            quantities will be included in ``arrays``; per-graph and per-edge
            quantities will be included in ``info``.

    Returns:
        A list of ``ase.Atoms`` objects if ``AtomicDataDict.BATCH_KEY`` is in self
        and is not None. Otherwise, a single ``ase.Atoms`` object is returned.
    """
    # === sanity check ===
    # exclude those that are special for ASE and that we process seperately
    special_handling_keys = [
        AtomicDataDict.POSITIONS_KEY,
        AtomicDataDict.CELL_KEY,
        AtomicDataDict.PBC_KEY,
        AtomicDataDict.ATOMIC_NUMBERS_KEY,
        AtomicDataDict.TOTAL_ENERGY_KEY,
        AtomicDataDict.FORCE_KEY,
        AtomicDataDict.PER_ATOM_ENERGY_KEY,
        AtomicDataDict.STRESS_KEY,
    ]
    assert (
        len(set(extra_fields).intersection(special_handling_keys)) == 0
    ), f"Cannot specify keys handled in special ways ({special_handling_keys}) as `extra_fields` for atoms output--- they are output by default"

    # == sort out logic for atomic numbers ==
    if AtomicDataDict.ATOMIC_NUMBERS_KEY in data:
        pass
    elif chemical_symbols is not None:
        atomic_num_to_atom_type_map = torch.tensor(
            [ase.data.atomic_numbers[spec] for spec in chemical_symbols],
            dtype=torch.int64,
            device="cpu",
        )
    else:
        warnings.warn(
            "Input data does not contain atomic numbers and `chemical_symbols` mapping to type index not provided ... using atom_type as atomic numbers instead, but this means the chemical symbols in ASE (outputs) will be wrong"
        )

    do_calc = any(
        k in data
        for k in [
            AtomicDataDict.TOTAL_ENERGY_KEY,
            AtomicDataDict.FORCE_KEY,
            AtomicDataDict.PER_ATOM_ENERGY_KEY,
            AtomicDataDict.STRESS_KEY,
        ]
    )

    # only select out fields that should be converted to ase
    fields = (
        special_handling_keys
        + extra_fields
        + [
            AtomicDataDict.ATOM_TYPE_KEY,
            AtomicDataDict.BATCH_KEY,
            AtomicDataDict.NUM_NODES_KEY,
        ]
    )
    data_pruned = {}
    for field in fields:
        if field in data:
            data_pruned[field] = data[field].to("cpu")

    atoms_list = []
    for idx in range(AtomicDataDict.num_frames(data_pruned)):
        # extract a single frame from the (possibly) batched data
        frame = AtomicDataDict.frame_from_batched(data_pruned, idx)

        if AtomicDataDict.ATOMIC_NUMBERS_KEY in frame:
            atomic_nums = frame[AtomicDataDict.ATOMIC_NUMBERS_KEY]
        elif chemical_symbols is not None:
            atomic_nums = torch.index_select(
                atomic_num_to_atom_type_map, 0, frame[AtomicDataDict.ATOM_TYPE_KEY]
            )
        else:
            atomic_nums = frame[AtomicDataDict.ATOM_TYPE_KEY]

        if AtomicDataDict.CELL_KEY in frame:
            cell = frame[AtomicDataDict.CELL_KEY].reshape((3, 3)).numpy()
        else:
            cell = None

        if AtomicDataDict.PBC_KEY in frame:
            pbc = frame[AtomicDataDict.PBC_KEY].reshape(-1).numpy()
        else:
            pbc = None

        mol = ase.Atoms(
            numbers=atomic_nums.reshape(-1).numpy(),
            positions=frame[AtomicDataDict.POSITIONS_KEY].numpy(),
            cell=cell,
            pbc=pbc,
        )

        if do_calc:
            fields = {}
            if AtomicDataDict.TOTAL_ENERGY_KEY in frame:
                fields["energy"] = (
                    frame[AtomicDataDict.TOTAL_ENERGY_KEY].reshape(-1).numpy()
                )
            if AtomicDataDict.PER_ATOM_ENERGY_KEY in frame:
                fields["energies"] = frame[AtomicDataDict.PER_ATOM_ENERGY_KEY].numpy()
            if AtomicDataDict.FORCE_KEY in frame:
                fields["forces"] = frame[AtomicDataDict.FORCE_KEY].numpy()
            if AtomicDataDict.STRESS_KEY in frame:
                # account for empty cell case, since model will provide empty stress entry
                if frame[AtomicDataDict.STRESS_KEY].numel() != 0:
                    fields["stress"] = full_3x3_to_voigt_6_stress(
                        frame[AtomicDataDict.STRESS_KEY].reshape((3, 3)).numpy()
                    )
            mol.calc = SinglePointCalculator(mol, **fields)

        # add other information
        for key in extra_fields:
            if key in _key_registry._NODE_FIELDS:
                # mask it
                mol.arrays[key] = frame[key].numpy()
            elif key in _key_registry._EDGE_FIELDS:
                mol.info[key] = frame[key].numpy()
            elif key == AtomicDataDict.EDGE_INDEX_KEY:
                mol.info[key] = frame[key].numpy()
            elif key in _key_registry._GRAPH_FIELDS:
                mol.info[key] = frame[key].reshape(-1).numpy()
            else:
                raise RuntimeError(
                    f"Extra field `{key}` isn't registered as node/edge/graph"
                )

        atoms_list.append(mol)

    return atoms_list
