from ._version import __version__  # noqa: F401

import packaging.version

import torch
import torchmetrics

# torch version checks
torch_version = packaging.version.parse(torch.__version__.split("+")[0])

# only allow 2.2.* or higher, required for `lightning` and `torchmetrics` compatibility
assert torch_version >= packaging.version.parse(
    "2.2"
), f"NequIP supports 2.2.* or later, but {torch_version} found"

# torchmetrics >= 1.6.0 for ddp autograd
# https://github.com/Lightning-AI/torchmetrics/releases/tag/v1.6.0
torchmetrics_version = packaging.version.parse(torchmetrics.__version__)
assert torchmetrics_version >= packaging.version.parse(
    "1.6.0"
), f"NequIP requires torchmetrics>=1.6.0 for ddp training but {torchmetrics_version} found"

# Load all installed nequip extension packages
# This allows installed extensions to register themselves in
# the nequip infrastructure with calls like `register_fields`

# see https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata
# we use "try ... except ..." to avoid importing sys.version_info
try:
    # python >= 3.10
    from importlib.metadata import entry_points

    _DISCOVERED_NEQUIP_EXTENSION = entry_points(group="nequip.extension")
except (ImportError, TypeError):
    # python < 3.10
    from importlib_metadata import entry_points

    _DISCOVERED_NEQUIP_EXTENSION = entry_points(group="nequip.extension")

for ep in _DISCOVERED_NEQUIP_EXTENSION:
    if ep.name == "init_always":
        ep.load()
