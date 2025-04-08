# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""
The functions in this script handle loading from saved formats, i.e. checkpoint files and package files (`.nequip.zip` files).
There are three main types of clients for these functions
1. users can interact with them through the config to load models
2. Python inference codes (e.g. ASE Calculator)
3. internal workflows, i.e. `nequip-package` and `nequip-compile`
"""

import torch

from .utils import (
    override_model_compile_mode,
    get_current_compile_mode,
    _COMPILE_MODE_OPTIONS,
    _EAGER_MODEL_KEY,
    _TRAIN_TIME_SCRIPT_KEY,
    _COMPILE_TIME_AOTINDUCTOR_KEY,
)
from nequip.scripts._workflow_utils import get_workflow_state
from nequip.utils import get_current_code_versions
from nequip.utils.logger import RankedLogger

import hydra
import warnings
from typing import List, Dict, Union, Any

# === setup logging ===
logger = RankedLogger(__name__, rank_zero_only=True)


def _check_compile_mode(compile_mode: str, client: str, exclude_keys: List[str] = []):
    # helper function for checking input arguments
    allowed_options = [
        mode for mode in _COMPILE_MODE_OPTIONS if mode not in exclude_keys
    ]
    assert (
        compile_mode in allowed_options
    ), f"`compile_mode={compile_mode}` is not recognized for `{client}`, only the following are supported: {allowed_options}"


def ModelFromCheckpoint(checkpoint_path: str, compile_mode: str = _EAGER_MODEL_KEY):
    """Builds model from a NequIP framework checkpoint file.

    This function can be used in the config file as follows.
    ::

      model:
        _target_: nequip.model.ModelFromCheckpoint
        checkpoint_path: path/to/ckpt
        compile_mode: eager/script/compile

    Args:
        checkpoint_path (str): path to a ``nequip`` framework checkpoint file
        compile_mode (str): ``eager``, ``script``, or ``compile`` allowed for training (note that ``script`` is not allowed if the checkpoint originates from a ``ModelFromPackage``)
    """
    # ^ there are other `compile_mode` options for internal use that are hidden from users
    exclude_modes = (
        [_COMPILE_TIME_AOTINDUCTOR_KEY] if get_workflow_state() == "train" else []
    )
    _check_compile_mode(compile_mode, "ModelFromCheckpoint", exclude_modes)
    logger.info(f"Loading model from checkpoint file: {checkpoint_path} ...")

    # === load checkpoint and extract info ===
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # === versions ===
    ckpt_versions = checkpoint["hyper_parameters"]["info_dict"]["versions"]
    session_versions = get_current_code_versions(verbose=False)

    for code, session_version in session_versions.items():
        if code in ckpt_versions:
            ckpt_version = ckpt_versions[code]
            # sanity check that versions for current build matches versions from ckpt
            if ckpt_version != session_version:
                warnings.warn(
                    f"`{code}` versions differ between the checkpoint file ({ckpt_version}) and the current run ({session_version}) -- `ModelFromCheckpoint` will be built with the current run's versions, but please check that this decision is as intended."
                )

    # === load model via lightning module ===
    training_module = hydra.utils.get_class(
        checkpoint["hyper_parameters"]["info_dict"]["training_module"]["_target_"]
    )
    # ensure that model is built with correct `compile_mode`
    with override_model_compile_mode(compile_mode):
        lightning_module = training_module.load_from_checkpoint(checkpoint_path)

    model = lightning_module.evaluation_model
    return model


# most of the complexity for `ModelFromPackage` is due to the need to keep track of the `Importer` if we ever repackage
# see `nequip/scripts/package.py` to get the full picture of how they interact

# main client `nequip-package` is expected to manipulate this dict via `_get_packaged_models`
_PACKAGED_MODELS: Dict[str, Union[torch.nn.Module, torch.package.PackageImporter]] = {}


def _get_packaged_models():
    global _PACKAGED_MODELS
    return _PACKAGED_MODELS


def ModelFromPackage(package_path: str, compile_mode: str = _EAGER_MODEL_KEY):
    """Builds model from a NequIP framework packaged zip file constructed with ``nequip-package``.

    This function can be used in the config file as follows.
    ::

      model:
        _target_: nequip.model.ModelFromPackage
        checkpoint_path: path/to/pkg
        compile_mode: eager/compile

    .. warning::
        Refrain from moving the package file if this model loader is used for training. Any process that loads a checkpoint produced from training runs originating from a package file will look for the original package file at the location specified during training.

    Args:
        package_path (str): path to NequIP framework packaged model with the ``.nequip.zip`` extension (an error will be thrown if the file has a different extension)
        compile_mode (str): ``eager`` or ``compile`` allowed for training
    """
    # ^ there are other `compile_mode` options for internal use that are hidden from users
    workflow_state = get_workflow_state()

    # === sanity check file extension ===
    assert str(package_path).endswith(
        ".nequip.zip"
    ), f"NequIP framework packaged files must have the `.nequip.zip` extension but found {str(package_path)}"

    # === account for checkpoint loading ===
    # if `ModelFromPackage` is used by itself, `override=False` and the input `compile_mode` argument is used
    # if this function is called at the end of checkpoint loading via `ModelFromCheckpoint`, `override=True` and the overriden `compile_mode` takes precedence
    cm, override = get_current_compile_mode(return_override=True)
    compile_mode = cm if override else compile_mode

    # === sanity check compile modes ===
    exclude_modes = [_COMPILE_TIME_AOTINDUCTOR_KEY] if workflow_state == "train" else []
    exclude_modes += [_TRAIN_TIME_SCRIPT_KEY]
    _check_compile_mode(compile_mode, "ModelFromPackage", exclude_modes)

    # === load model ===
    logger.info(f"Loading model from package file: {package_path} ...")
    with warnings.catch_warnings():
        # suppress torch.package TypedStorage warning
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated.*",
            category=UserWarning,
            module="torch.package.package_importer",
        )
        imp = torch.package.PackageImporter(package_path)

        # load packaging metadata that can be used to condition loading logic
        pkg_metadata: Dict[str, Any] = imp.load_pickle(
            package="model", resource="package_metadata.pkl"
        )
        available_models = pkg_metadata["available_models"]
        # throw warning if desired `compile_mode` is not available, and default to eager
        if compile_mode not in available_models:
            warnings.warn(
                f"Requested `{compile_mode}` model is not present in the package file ({package_path}). `nequip-{workflow_state}` task will default to using the `_EAGER_MODEL_KEY` model."
            )
            compile_mode = _EAGER_MODEL_KEY

        # if we're loading for `nequip-package` (through `ModelFromCheckpoint`)
        # we load every model type and save it, along with the importer
        # `nequip-package` will query `_PACKAGED_MODELS` and handle the necessary logic including
        # 1. loading the state dict of the model returned by this function (that will get updated by the checkpoint), into the other model types
        # 2. handle the repackaging logic (https://pytorch.org/docs/stable/package.html#re-export-an-imported-object)
        if workflow_state == "package":
            global _PACKAGED_MODELS
            _PACKAGED_MODELS.update({"importer": imp})
            # note that the loop is over `available_models`
            for mode in available_models:
                model = imp.load_pickle(
                    package="model",
                    resource=f"{mode}_model.pkl",
                    map_location="cpu",
                )
                _PACKAGED_MODELS.update({mode: model})
            model = _PACKAGED_MODELS[compile_mode]
        else:
            model = imp.load_pickle(
                package="model",
                resource=f"{compile_mode}_model.pkl",
                map_location="cpu",
            )

    # NOTE: model returned is not a GraphModel object tied to the `nequip` in current Python env, but a GraphModel object from the packaged zip file
    return model
