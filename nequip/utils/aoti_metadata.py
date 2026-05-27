# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import importlib
import zipfile
from typing import List, Final, Set

NEQUIP_AOTI_INPUTS_KEY: Final[str] = "nequip_aoti_inputs"
NEQUIP_AOTI_OUTPUTS_KEY: Final[str] = "nequip_aoti_outputs"
NEQUIP_CUSTOM_OPS_LIBS_KEY: Final[str] = "nequip_custom_ops_libs"

_CUSTOM_OPS_LIBS_ENTRY = "nequip_custom_ops_libs.txt"


def serialize_aoti_keys(keys: List[str]) -> str:
    assert all(" " not in key for key in keys), (
        f"AOTI field names cannot contain spaces: {keys}"
    )
    return " ".join(keys)


def parse_aoti_keys(serialized_keys: str) -> List[str]:
    return serialized_keys.split()


def embed_custom_ops_libs(pt2_path: str, custom_ops_libs: Set[str]) -> None:
    """Append a custom ops libs entry to an existing AOTI .pt2 zip archive.

    Called after ``aoti_compile_and_package`` to record which Python libraries must be imported before the package can be loaded.
    The entry is written as a plain space-separated text file so that ``import_custom_ops_libs`` can read it with a bare ``zipfile`` open before PyTorch's C++ loader runs.

    Args:
        pt2_path: path to the .pt2 file to modify in-place.
        custom_ops_libs: set of importable library names (e.g. ``{"openequivariance"}``).
    """
    if not custom_ops_libs:
        return
    with zipfile.ZipFile(pt2_path, "a") as zf:
        zf.writestr(_CUSTOM_OPS_LIBS_ENTRY, " ".join(sorted(custom_ops_libs)))


def import_custom_ops_libs(pt2_path: str) -> None:
    """Read the custom ops libs entry from a .pt2 archive and import each library.

    Must be called *before* ``torch._inductor.aoti_load_package`` so that custom op schemas are registered before the C++ ``AOTIModelPackageLoader`` runs.

    No-op if the entry is absent (e.g. models compiled without custom ops, or models compiled before this feature was added).

    Args:
        pt2_path: path to the .pt2 file to inspect.
    """
    with zipfile.ZipFile(pt2_path, "r") as zf:
        if _CUSTOM_OPS_LIBS_ENTRY not in zf.namelist():
            return
        for lib in zf.read(_CUSTOM_OPS_LIBS_ENTRY).decode().split():
            importlib.import_module(lib)
