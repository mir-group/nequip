# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""
Functions for loading models from package files.
"""

import torch
import yaml
import warnings
import contextlib
from typing import Dict, Any

from nequip.data import AtomicDataDict
from nequip.model.utils import (
    get_current_compile_mode,
    _EAGER_MODEL_KEY,
)
from nequip.scripts._workflow_utils import get_workflow_state
from nequip.utils.logger import RankedLogger

from ._utils import _check_compile_mode, _check_file_exists


# === setup logging ===
logger = RankedLogger(__name__, rank_zero_only=True)

# === package importer utilities ===
# most of the complexity for `ModelFromPackage` is due to the need to keep track of the `Importer` if we ever repackage
# see `nequip/scripts/package.py` to get the full picture of how they interact
# we expect the following variable to only be used during `nequip-package`

_PACKAGE_TIME_SHARED_IMPORTER = None


def _get_shared_importer():
    global _PACKAGE_TIME_SHARED_IMPORTER
    return _PACKAGE_TIME_SHARED_IMPORTER


def _get_package_metadata(imp) -> Dict[str, Any]:
    """Load packaged model metadata from an existing PackageImporter."""
    pkg_metadata: Dict[str, Any] = yaml.safe_load(
        imp.load_text(package="model", resource="package_metadata.txt")
    )
    assert int(pkg_metadata["package_version_id"]) > 0
    # ^ extra sanity check since saving metadata in txt files was implemented in packaging version 1

    return pkg_metadata


# === warning management ===


@contextlib.contextmanager
def _suppress_package_importer_warnings():
    # Ideally this ceases to exist or becomes a no-op in future versions of PyTorch
    with warnings.catch_warnings():
        # suppress torch.package TypedStorage warning
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated.*",
            category=UserWarning,
            module="torch.package.package_importer",
        )
        yield


# === loading models from package files ===


def ModelFromPackage(package_path: str, compile_mode: str = _EAGER_MODEL_KEY):
    """Builds model from a NequIP framework packaged zip file constructed with ``nequip-package``.

    This function can be used in the config file as follows.

    .. code-block:: yaml

      model:
        _target_: nequip.model.ModelFromPackage
        package_path: path/to/pkg
        compile_mode: eager/compile

    .. warning::
        DO NOT CHANGE the directory structure or location of the package file if this model loader is used for training. Any process that loads a checkpoint produced from training runs originating from a package file will look for the original package file at the location specified during training. It is also recommended to use full paths (instead or relative paths) to avoid potential errors.

    Args:
        package_path (str): path to NequIP framework packaged model with the ``.nequip.zip`` extension (an error will be thrown if the file has a different extension)
        compile_mode (str): ``eager`` or ``compile`` allowed for training
    """
    # === sanity checks ===
    _check_file_exists(file_path=package_path, file_type="package")
    assert str(package_path).endswith(
        ".nequip.zip"
    ), f"NequIP framework packaged files must have the `.nequip.zip` extension but found {str(package_path)}"

    # === account for checkpoint loading ===
    # if `ModelFromPackage` is used by itself, `override=False` and the input `compile_mode` argument is used
    # if this function is called at the end of checkpoint loading via `ModelFromCheckpoint`, `override=True` and the overriden `compile_mode` takes precedence
    cm, override = get_current_compile_mode(return_override=True)
    compile_mode = cm if override else compile_mode

    # === sanity check compile modes ===
    workflow_state = get_workflow_state()
    _check_compile_mode(compile_mode, "ModelFromPackage")

    # === load model ===
    logger.info(f"Loading model from package file: {package_path} ...")
    with _suppress_package_importer_warnings():
        # during `nequip-package`, we need to use the same importer for all the models for successful repackaging
        # see https://pytorch.org/docs/stable/package.html#re-export-an-imported-object
        if workflow_state == "package":
            global _PACKAGE_TIME_SHARED_IMPORTER
            imp = _PACKAGE_TIME_SHARED_IMPORTER
            # we load the importer from `package_path` for the first time
            if imp is None:
                imp = torch.package.PackageImporter(package_path)
                _PACKAGE_TIME_SHARED_IMPORTER = imp
            # if it's not `None`, it means we've previously loaded a model during `nequip-package` and should keep using the same importer
        else:
            # if not doing `nequip-package`, we just load a new importer every time `ModelFromPackage` is called
            imp = torch.package.PackageImporter(package_path)

        # do sanity checking with available models
        pkg_metadata = _get_package_metadata(imp)
        available_models = pkg_metadata["available_models"]
        # throw warning if desired `compile_mode` is not available, and default to eager
        if compile_mode not in available_models:
            warnings.warn(
                f"Requested `{compile_mode}` model is not present in the package file ({package_path}). `nequip-{workflow_state}` task will default to using the `{_EAGER_MODEL_KEY}` model."
            )
            compile_mode = _EAGER_MODEL_KEY

        model = imp.load_pickle(
            package="model",
            resource=f"{compile_mode}_model.pkl",
            map_location="cpu",
        )

    # NOTE: model returned is not a GraphModel object tied to the `nequip` in current Python env, but a GraphModel object from the packaged zip file
    return model


def data_dict_from_package(package_path: str) -> AtomicDataDict.Type:
    """Load example data from a .nequip.zip package file."""
    with _suppress_package_importer_warnings():
        imp = torch.package.PackageImporter(package_path)
        data = imp.load_pickle(package="model", resource="example_data.pkl")
    return data
