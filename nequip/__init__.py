import sys

from ._version import __version__  # noqa: F401

import packaging.version

import torch

# torch version checks
torch_version = packaging.version.parse(torch.__version__)

# only allow 1.13* or higher
assert torch_version >= packaging.version.parse(
    "1.13"
), f"NequIP supports 1.13.* or later, but {torch_version} found"


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
