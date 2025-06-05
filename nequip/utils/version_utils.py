# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
# Separate version utilities to avoid circular imports with __init__.py
import importlib.metadata
from typing import Optional

_ALL_PKGS = importlib.metadata.packages_distributions()


def get_version_safe(module_name: str) -> Optional[str]:
    """Safely get the version of an installed package based on its module name.

    Args:
        module_name: name of the module to get version for

    Returns:
        version string if package is found, None otherwise
    """
    try:
        if module_name in _ALL_PKGS:
            module_name = _ALL_PKGS[module_name][0]
        return importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        return None
