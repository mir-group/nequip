"""
`nequip-package` generates the archival format for NequIP models. This file contains the information necessary to track the archival format itself.

Whenever the archival format changes, `_CURRENT_NEQUIP_PACKAGE_VERSION` (counter to track the packaged model format) should be bumped up to the next number. We can then condition `ModelFromPackage` on the packaging format version to decide code paths to load the model appropriately.

We also track `_PACKAGE_MODEL_TYPE_DICT`, which determines the set of models to be built and packaged at the time of packaging. Clients can then decide which packaged model to load for their desired task.
"""

from typing import Final, Dict, Any

# `nequip-package` format version index to condition other features upon when loading `nequip-package` from a specific version
_CURRENT_NEQUIP_PACKAGE_VERSION = 0

# we package several different model types depending on what the packaged model will eventually be used for

# basic eager PyTorch model
_EAGER_MODEL_KEY = "eager"
# model that will be used for `nequip-compile` with AOT Inductor
_AOTINDUCTOR_MODEL_KEY = "aotinductor"

_PACKAGE_MODEL_TYPE_DICT: Final[Dict[str, Any]] = {
    _EAGER_MODEL_KEY: None,
    _AOTINDUCTOR_MODEL_KEY: "aotinductor",
}
