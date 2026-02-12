# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

"""Custom OmegaConf resolvers for nequip."""

import logging
from omegaconf import OmegaConf, ListConfig
from typing import Dict, Callable, Set, Any, Optional

from nequip.utils import get_project_root

logger = logging.getLogger(__name__)


def _sanitize_int(x, client: str):
    err_msg = f"`{client} resolver accepts nonnegative integer inputs, but found {x}"
    if isinstance(x, str):
        assert x.isdigit(), err_msg
        x = int(x)
    assert isinstance(x, int), err_msg
    return x


def int_div(a, b):
    """Integer division resolver for OmegaConf."""
    a = _sanitize_int(a, "int_div")
    b = _sanitize_int(b, "int_div")

    if a % b != 0:
        raise ValueError(
            f"`int_div` requires exact division, but {a} is not divisible by {b}"
        )

    return a // b


def int_mul(a, b):
    """Integer multiplication resolver for OmegaConf."""
    a = _sanitize_int(a, "int_mul")
    b = _sanitize_int(b, "int_mul")
    return a * b


def float_to_str(x: float, fmt: str = ".1f") -> str:
    """Format a float to string using a given format."""
    if not fmt.startswith("."):
        raise ValueError(f"Format string must start with '.', got '{fmt}'")
    return format(x, fmt)


def concat_lists(list1, list2):
    """Concatenate two lists."""
    if not isinstance(list1, (list, tuple, ListConfig)):
        raise ValueError(
            f"First argument must be a list, tuple, or ListConfig, got {type(list1)}"
        )
    if not isinstance(list2, (list, tuple, ListConfig)):
        raise ValueError(
            f"Second argument must be a list, tuple, or ListConfig, got {type(list2)}"
        )
    return list1 + list2


def list_to_identity_dict(x):
    """Convert a list to an identity dictionary mapping each element to itself."""
    if not isinstance(x, (list, tuple, ListConfig)):
        raise ValueError(
            f"Argument must be a list, tuple, or ListConfig, got {type(x)}"
        )
    return {item: item for item in x}


def list_to_constant_dict(keys, value):
    """Convert a list to a dictionary mapping each element to a constant value."""
    if not isinstance(keys, (list, tuple, ListConfig)):
        raise ValueError(
            f"First argument must be a list, tuple, or ListConfig, got {type(keys)}"
        )
    return {key: value for key in keys}


def big_dataset_stats(name: str, cutoff_radius: float) -> Dict[str, Any]:
    """Get precomputed dataset statistics for large datasets."""
    root = get_project_root()
    stats_path = root / "data" / "dataset_stats" / f"{name}.yaml"
    if not stats_path.exists():
        raise ValueError(f"No precomputed dataset stats for dataset '{name}'")

    stats = OmegaConf.load(stats_path)
    cutoff_radius = float_to_str(cutoff_radius)
    # Retrieve the stats for the given cutoff radius
    stats["num_neighbors_mean"] = stats["num_neighbors_mean"].get(cutoff_radius, None)
    if stats["num_neighbors_mean"] is None:
        raise ValueError(
            f"No precomputed dataset stats for dataset '{name}' with cutoff radius {cutoff_radius} (tried key '{cutoff_radius}')"
        )
    stats["per_type_num_neighbors_mean"] = stats["per_type_num_neighbors_mean"].get(
        cutoff_radius, None
    )
    if stats["per_type_num_neighbors_mean"] is None:
        raise ValueError(
            f"No precomputed dataset stats for dataset '{name}' with cutoff radius {cutoff_radius} (tried key '{cutoff_radius}')"
        )

    return stats


def type_names_from_package(package_path: str):
    """Extract type names from a packaged model file."""
    from nequip.model import ModelTypeNamesFromPackage

    return ModelTypeNamesFromPackage(package_path)


def cutoff_radius_from_package(
    package_path: str, model_key: Optional[str] = None
) -> float:
    """Extract cutoff radius (r_max) from a packaged model file.

    For packages containing multiple models (ModuleDict), ``model_key`` can be
    provided to select which model to inspect.
    """
    from nequip.model.saved_models import load_saved_model
    from nequip.nn import graph_model
    from nequip.train.lightning import _SOLE_MODEL_KEY

    if model_key is None:
        model_key = _SOLE_MODEL_KEY
    model = load_saved_model(
        package_path,
        compile_mode="eager",
        model_key=model_key,
    )
    r_max = float(model.metadata[graph_model.R_MAX_KEY])
    logger.info(
        f"Extracted cutoff radius {r_max} from path {package_path} (model_key={model_key})"
    )
    return r_max


# === Resolver Registry ===

_DEFAULT_RESOLVERS: Dict[str, Callable] = {
    "int_div": int_div,
    "int_mul": int_mul,
    "concat_lists": concat_lists,
    "list_to_identity_dict": list_to_identity_dict,
    "list_to_constant_dict": list_to_constant_dict,
    "big_dataset_stats": big_dataset_stats,
    "type_names_from_package": type_names_from_package,
    "cutoff_radius_from_package": cutoff_radius_from_package,
}

_REGISTERED_RESOLVERS: Set[str] = set()
_DEFAULT_RESOLVERS_REGISTERED: bool = False


def register_resolvers(resolvers: Dict[str, Callable]) -> None:
    """Register custom OmegaConf resolvers.

    Args:
        resolvers (Dict[str, Callable]): mapping from resolver name to resolver function

    Raises:
        ValueError: if any resolver name conflicts with existing registered resolvers
    """
    # Check for conflicts with already registered resolvers
    new_names = set(resolvers.keys())
    conflicts = new_names.intersection(_REGISTERED_RESOLVERS)
    if conflicts:
        raise ValueError(
            f"Resolver name(s) {conflicts} already registered. Cannot register the same resolver name twice."
        )

    # Register each resolver with OmegaConf
    for name, func in resolvers.items():
        OmegaConf.register_new_resolver(name, func)
        _REGISTERED_RESOLVERS.add(name)


def _register_default_resolvers():
    """Register all default nequip OmegaConf resolvers."""
    global _DEFAULT_RESOLVERS_REGISTERED
    if not _DEFAULT_RESOLVERS_REGISTERED:
        register_resolvers(_DEFAULT_RESOLVERS)
        _DEFAULT_RESOLVERS_REGISTERED = True
