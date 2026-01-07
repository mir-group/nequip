# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

from typing import List, Dict, Union
import torch

from nequip.utils.global_dtype import _GLOBAL_DTYPE


# conversion flow: partial_dict -> full_dict -> tensor -> str
#                                                           |
#                                                           v
#                                                       full_dict


def cutoff_partialdict_to_fulldict(
    partial_dict: Dict[str, Union[float, Dict[str, float]]],
    type_names: List[str],
    r_max: float,
) -> Dict[str, Dict[str, float]]:
    """Convert partial cutoff dict to full dict with all entries.

    Fills missing entries with ``r_max``.

    Args:
        partial_dict: partial specification from config,
            e.g. ``{"H": 2.0, "C": {"H": 4.0, "C": 3.5}}``
        type_names: list of atom type names
        r_max: global cutoff radius (default for missing entries)

    Returns:
        full dict with all source -> target pairs specified,
        e.g. ``{"H": {"H": 2.0, "C": 2.0}, "C": {"H": 4.0, "C": 3.5}}``
    """
    full_dict = {}
    for source_type in type_names:
        full_dict[source_type] = {}
        if source_type in partial_dict:
            entry = partial_dict[source_type]
            if isinstance(entry, float):
                # uniform cutoff for this source type
                for target_type in type_names:
                    full_dict[source_type][target_type] = entry
            else:
                # per-target specification
                for target_type in type_names:
                    if target_type in entry:
                        full_dict[source_type][target_type] = entry[target_type]
                    else:
                        # missing target defaults to r_max
                        full_dict[source_type][target_type] = r_max
        else:
            # missing source defaults to r_max for all targets
            for target_type in type_names:
                full_dict[source_type][target_type] = r_max

    return full_dict


def cutoff_fulldict_to_tensor(
    full_dict: Dict[str, Dict[str, float]],
    type_names: List[str],
) -> torch.Tensor:
    """Convert full cutoff dict to tensor.

    Args:
        full_dict: full specification with all source -> target pairs
        type_names: list of atom type names

    Returns:
        tensor of shape ``(num_types, num_types)`` with per-edge-type cutoffs
    """
    num_types = len(type_names)
    cutoff_list = []
    for source_type in type_names:
        row = []
        for target_type in type_names:
            row.append(full_dict[source_type][target_type])
        cutoff_list.append(row)

    cutoff_tensor = torch.as_tensor(cutoff_list, dtype=_GLOBAL_DTYPE).contiguous()
    assert cutoff_tensor.shape == (num_types, num_types)
    assert torch.all(cutoff_tensor > 0)
    return cutoff_tensor


def cutoff_tensor_to_str(cutoff_tensor: torch.Tensor) -> str:
    """Convert tensor to metadata string format.

    Args:
        cutoff_tensor: cutoff values as tensor (any shape, will be flattened)

    Returns:
        space-separated string of cutoff values in row-major order
    """
    return " ".join(str(r.item()) for r in cutoff_tensor.reshape(-1))


def cutoff_str_to_fulldict(
    cutoff_str: str,
    type_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Convert metadata string to full dict format.

    Args:
        cutoff_str: space-separated string of cutoff values
        type_names: list of atom type names

    Returns:
        full dict with all source -> target pairs specified
    """
    if cutoff_str in ("", None):
        return None

    cutoff_values = [float(x) for x in cutoff_str.split()]
    num_types = len(type_names)

    assert len(cutoff_values) == num_types * num_types, (
        f"Expected {num_types * num_types} cutoff values, got {len(cutoff_values)}"
    )

    full_dict = {}
    for i, source_type in enumerate(type_names):
        full_dict[source_type] = {}
        for j, target_type in enumerate(type_names):
            full_dict[source_type][target_type] = cutoff_values[i * num_types + j]

    return full_dict


def cutoff_partialdict_to_tensor(
    partial_dict: Dict[str, Union[float, Dict[str, float]]],
    type_names: List[str],
    r_max: float,
) -> torch.Tensor:
    """Composes ``cutoff_partialdict_to_fulldict`` and ``cutoff_fulldict_to_tensor``."""
    full_dict = cutoff_partialdict_to_fulldict(partial_dict, type_names, r_max)
    cutoff_tensor = cutoff_fulldict_to_tensor(full_dict, type_names)
    assert torch.all(cutoff_tensor <= r_max)
    return cutoff_tensor


def cutoff_partialdict_to_str(
    partial_dict: Dict[str, Union[float, Dict[str, float]]],
    type_names: List[str],
    r_max: float,
) -> str:
    """Composes ``cutoff_partialdict_to_fulldict``, ``cutoff_fulldict_to_tensor``, and ``cutoff_tensor_to_str``."""
    full_dict = cutoff_partialdict_to_fulldict(partial_dict, type_names, r_max)
    tensor = cutoff_fulldict_to_tensor(full_dict, type_names)
    return cutoff_tensor_to_str(tensor)
