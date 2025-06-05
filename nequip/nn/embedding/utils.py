# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

from typing import List, Dict, Union


def per_edge_type_cutoff_to_metadata_str(
    type_names: List[str],
    per_edge_type_cutoff: Dict[str, Union[float, Dict[str, float]]],
    r_max: float,
) -> str:
    """Convert per-edge-type cutoff dict to flattened metadata string.

    Args:
        type_names: list of atom type names
        per_edge_type_cutoff: cutoff dict from config
        r_max: global cutoff radius

    Returns:
        space-separated string of cutoff values in row-major order
    """
    from ._edge import _process_per_edge_type_cutoff

    cutoff_tensor = _process_per_edge_type_cutoff(
        type_names, per_edge_type_cutoff, r_max
    )
    return " ".join(str(e.item()) for e in cutoff_tensor.view(-1))


def parse_per_edge_type_cutoff_metadata(
    cutoff_str: str, type_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Parse flattened per-edge-type cutoff metadata string to dict format.

    Reverse operation of `per_edge_type_cutoff_to_metadata_str`.

    Args:
        cutoff_str: space-separated string of cutoff values
        type_names: list of atom type names

    Returns:
        dict format suitable for NeighborListTransform
    """
    # short-circuit
    if cutoff_str in ("", None):
        return None

    cutoff_values = [float(x) for x in cutoff_str.split()]
    num_types = len(type_names)

    assert (
        len(cutoff_values) == num_types * num_types
    ), f"Expected {num_types * num_types} cutoff values, but got {len(cutoff_values)}"

    result = {}
    for i, source_type in enumerate(type_names):
        result[source_type] = {}
        for j, target_type in enumerate(type_names):
            result[source_type][target_type] = cutoff_values[i * num_types + j]

    return result
