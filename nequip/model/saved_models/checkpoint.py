# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""
Functions for loading models from checkpoint files.
"""

import torch
import hydra
import warnings

from nequip.model.utils import (
    override_model_compile_mode,
    _EAGER_MODEL_KEY,
)
from nequip.utils import get_current_code_versions
from nequip.utils.logger import RankedLogger

from ._utils import _check_compile_mode, _check_file_exists

# === setup logging ===
logger = RankedLogger(__name__, rank_zero_only=True)


def ModelFromCheckpoint(checkpoint_path: str, compile_mode: str = _EAGER_MODEL_KEY):
    """Builds model from a NequIP framework checkpoint file.

    This function can be used in the config file as follows.
    ::

      model:
        _target_: nequip.model.saved_models.ModelFromCheckpoint
        checkpoint_path: path/to/ckpt
        compile_mode: eager/compile

    .. warning::
        DO NOT CHANGE the directory structure or location of the checkpoint file if this model loader is used for training. Any process that loads a checkpoint produced from training runs originating from a package file will look for the original package file at the location specified during training. It is also recommended to use full paths (instead or relative paths) to avoid potential errors.

    Args:
        checkpoint_path (str): path to a ``nequip`` framework checkpoint file
        compile_mode (str): ``eager`` or ``compile`` allowed for training
    """
    # === sanity checks ===
    _check_file_exists(file_path=checkpoint_path, file_type="checkpoint")
    _check_compile_mode(compile_mode, "ModelFromCheckpoint")
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
