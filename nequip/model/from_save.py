import torch
from nequip.utils import get_current_code_versions

import hydra
import warnings


def ModelFromCheckpoint(checkpoint_path: str):
    """Builds model from a ``nequip`` framework checkpoint file.

    This model builder is intended for training a pre-trained model from a checkpoint file.

    The behavior of this model builder is such that it
    - will ignore the global options from the checkpoint file.
    - will use the compile mode from the checkpoint (unless whoever calls this uses the compile_mode overriding context manager)

    The entity calling this model builder is responsible for setting up the global options and potentially overriding the compile mode.

    Args:
        checkpoint_path (str): path to a ``nequip`` framework checkpoint file
    """
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
        ckpt_version = ckpt_versions[code]
        # sanity check that versions for current build matches versions from ckpt
        # TODO: or should we just throw an error
        if ckpt_version != session_version:
            warnings.warn(
                f"`{code}` versions differ between the checkpoint file ({ckpt_version}) and the current run ({session_version}) -- current model will be built with the versions, check that this is the intended behavior"
            )

    # === load model via lightning module ===
    training_module = hydra.utils.get_class(
        checkpoint["hyper_parameters"]["info_dict"]["training_module"]["_target_"]
    )
    lightning_module = training_module.load_from_checkpoint(checkpoint_path)
    return lightning_module.model


def ModelFromPackage(package_path: str):
    """Builds model from a packaged zip file (with ``nequip-package``).

    Args:
        package_path (str): path to packaged model (a zip file)
    """
    with warnings.catch_warnings():
        # suppress torch.package TypedStorage warning
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated.*",
            category=UserWarning,
            module="torch.package.package_importer",
        )
        imp = torch.package.PackageImporter(package_path)
        model = imp.load_pickle(package="model", resource="model.pkl")
    return model
