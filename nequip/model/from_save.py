import torch
from nequip.utils import get_current_code_versions
from nequip.utils._global_options import _set_global_options

from omegaconf import OmegaConf
import hydra
import warnings
from typing import Union


def ModelFromCheckpoint(
    checkpoint_path: str,
    set_global_options: Union[str, bool] = False,
):
    """Builds model from a NequIP framework checkpoint file.

    There are two common use modes of this model builder.

      1. In ``nequip-train``, ``ModelFromCheckpoint`` can be used to train, validate and/or test a pre-trained model from a checkpoint file.
      2. In a Python script, ``ModelFromCheckpoint`` can be used to load a model from a checkpoint file for custom evaluation tasks.

    For use in custom Python scripts, note that this model builder

      - will ignore the global options from the checkpoint file, and
      - will use the ``compile_mode`` from the checkpoint (unless whoever calls this uses the ``compile_mode`` overriding context manager)
    The script using this model builder is responsible for setting up the global options and potentially overriding the compile mode.

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
                f"`{code}` versions differ between the checkpoint file ({ckpt_version}) and the current run ({session_version}) -- `ModelFromCheckpoint` will be built with the current run's versions, but please check that this decision is as intended."
            )

    if set_global_options:
        global_options = checkpoint["hyper_parameters"]["info_dict"]["global_options"]
        global_options.update({"warn_on_override": set_global_options == "warn"})
        _set_global_options(**global_options)

    # === load model via lightning module ===
    training_module = hydra.utils.get_class(
        checkpoint["hyper_parameters"]["info_dict"]["training_module"]["_target_"]
    )
    lightning_module = training_module.load_from_checkpoint(checkpoint_path)
    return lightning_module.model


def ModelFromPackage(
    package_path: str,
    set_global_options: Union[str, bool] = False,
):
    """Builds model from a NequIP framework packaged zip file constructed with ``nequip-package``.

    Args:
        package_path (str): path to NequIP framework packaged model with the ``.nequip.zip`` extension (an error will be thrown if the file has a different extension)
    """
    assert str(package_path).endswith(
        ".nequip.zip"
    ), f"NequIP framework packaged files must have the `.nequip.zip` extension but found {str(package_path)}"
    with warnings.catch_warnings():
        # suppress torch.package TypedStorage warning
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated.*",
            category=UserWarning,
            module="torch.package.package_importer",
        )
        imp = torch.package.PackageImporter(package_path)
        # set global options before loading model
        # TODO: do we need to set global options with the packaged code?
        # for now, nothing in global options should require this, but maybe if e3nn were interned ...
        if set_global_options:
            config = imp.load_text(package="model", resource="config.yaml")
            global_options = OmegaConf.to_container(OmegaConf.create(config))[
                "global_options"
            ]
            global_options.update({"warn_on_override": set_global_options == "warn"})
            _set_global_options(**global_options)
        model = imp.load_pickle(package="model", resource="model.pkl")

    # NOTE: model returned is not a GraphModel object tied to the `nequip` in current Python env, but a GraphModel object from the packaged zip file
    return model
