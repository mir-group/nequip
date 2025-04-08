# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from nequip.data import AtomicDataDict
from typing import Dict, List, Callable, Union

# === Inputs and Outputs for AOT Compile ===
# standard sets of input and output fields for specific integrations

PAIR_NEQUIP_INPUTS = [
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
    AtomicDataDict.CELL_KEY,
    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
]

LMP_OUTPUTS = [
    AtomicDataDict.PER_ATOM_ENERGY_KEY,
    AtomicDataDict.FORCE_KEY,
    AtomicDataDict.VIRIAL_KEY,
]

ASE_OUTPUTS = [
    AtomicDataDict.PER_ATOM_ENERGY_KEY,
    AtomicDataDict.TOTAL_ENERGY_KEY,
    AtomicDataDict.FORCE_KEY,
    AtomicDataDict.STRESS_KEY,
]


# === batch map rules ===
def single_frame_batch_map_settings(batch_map):
    # make num_frames batch dims static, for single frame case
    # relevant for single-frame use cases, e.g. pair_nequip and ase
    batch_map["graph"] = torch.export.Dim.STATIC
    return batch_map


# === data rules ===
def single_frame_data_settings(data):
    # because of the 0/1 specialization problem,
    # and the fact that the LAMMPS pair style (and ASE) requires `num_frames=1`
    # we need to augment to data to remove the `BATCH_KEY` and `NUM_NODES_KEY`
    # to take more optimized code paths
    if AtomicDataDict.BATCH_KEY in data:
        data.pop(AtomicDataDict.BATCH_KEY)
        data.pop(AtomicDataDict.NUM_NODES_KEY)
    return data


PAIR_NEQUIP_TARGET = {
    "input": PAIR_NEQUIP_INPUTS,
    "output": LMP_OUTPUTS,
    "batch_map_settings": single_frame_batch_map_settings,
    "data_settings": single_frame_data_settings,
}
ASE_TARGET = {
    "input": PAIR_NEQUIP_INPUTS,
    "output": ASE_OUTPUTS,
    "batch_map_settings": single_frame_batch_map_settings,
    "data_settings": single_frame_data_settings,
}


COMPILE_TARGET_DICT = {
    "pair_nequip": PAIR_NEQUIP_TARGET,
    "ase": ASE_TARGET,
}


def register_compile_targets(
    target_dict: Dict[str, Union[List[str], Callable]],
) -> None:
    """Register compile targets for AOT compilation.

    The intended clients of this function are NequIP extension packages to register their custom compilation targets.

    Args:
        target_dict: dict containing keys `input`, `output`, `batch_map_settings`, `data_settings`
    """
    # update target dict
    global COMPILE_TARGET_DICT
    COMPILE_TARGET_DICT.update(target_dict)
