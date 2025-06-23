from ._version import __version__  # noqa: F401

import packaging.version

import torch

from .utils.resolvers import _register_default_resolvers
from .utils.versions.version_utils import get_version_safe


# torch version checks
torch_version = packaging.version.parse(get_version_safe(torch.__name__).split("+")[0])

# only allow 2.2.* or higher, required for `lightning` and `torchmetrics` compatibility
assert torch_version >= packaging.version.parse(
    "2.2"
), f"NequIP supports 2.2.* or later, but {torch_version} found"

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

# register OmegaConf resolvers
_register_default_resolvers()
