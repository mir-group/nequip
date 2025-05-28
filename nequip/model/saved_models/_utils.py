# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""
Shared utilities for loading models from saved formats (checkpoints and packages).
"""

import os
from typing import List

from nequip.model.utils import _COMPILE_MODE_OPTIONS


def _check_compile_mode(compile_mode: str, client: str, exclude_keys: List[str] = []):
    """Helper function for checking input arguments."""
    allowed_options = [
        mode for mode in _COMPILE_MODE_OPTIONS if mode not in exclude_keys
    ]
    assert (
        compile_mode in allowed_options
    ), f"`compile_mode={compile_mode}` is not recognized for `{client}`, only the following are supported: {allowed_options}"


def _check_file_exists(file_path: str, file_type: str):
    """Check if a checkpoint or package file exists."""
    if not os.path.isfile(file_path):
        assert file_type in ("checkpoint", "package")
        client = (
            "`ModelFromCheckpoint`"
            if file_type == "checkpoint"
            else "`ModelFromPackage`"
        )
        raise RuntimeError(
            f"{file_type} file provided at `{file_path}` is not found. NOTE: Any process that loads a checkpoint produced from training runs based on {client} will look for the original {file_type} file at the location specified during training. It is also recommended to use full paths (instead or relative paths) to avoid potential errors."
        )
