import os
import sys

from ._version import __version__  # noqa: F401

import packaging.version

import torch
import warnings

# torch version checks
torch_version = packaging.version.parse(torch.__version__)

# only allow 1.11*, 1.13* or higher (no 1.12.*)
assert (torch_version == packaging.version.parse("1.11")) or (
    torch_version >= packaging.version.parse("1.13")
), f"NequIP supports PyTorch 1.11.* or 1.13.* or later, but {torch_version} found"

# warn if using 1.13* or 2.0.*
if (
    packaging.version.parse("1.13.0") <= torch_version
    and int(os.environ.get("PYTORCH_VERSION_WARNING", 1)) != 0
):
    warnings.warn(
        f"!! PyTorch version {torch_version} found. Upstream issues in PyTorch versions 1.13.* and 2.* have been seen to cause unusual performance degredations on some CUDA systems that become worse over time; see https://github.com/mir-group/nequip/discussions/311. The best tested PyTorch version to use with CUDA devices is 1.11; while using other versions if you observe this problem, an unexpected lack of this problem, or other strange behavior, please post in the linked GitHub issue."
    )


# Load all installed nequip extension packages
# This allows installed extensions to register themselves in
# the nequip infrastructure with calls like `register_fields`

# see https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

_DISCOVERED_NEQUIP_EXTENSION = entry_points(group="nequip.extension")
for ep in _DISCOVERED_NEQUIP_EXTENSION:
    if ep.name == "init_always":
        ep.load()
