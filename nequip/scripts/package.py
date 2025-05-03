# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import argparse

import pathlib
import yaml
import warnings

# TODO: check if we still need this?
# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401
import torch

from nequip.data.compile_utils import data_dict_from_checkpoint
from nequip.model import ModelFromCheckpoint
from nequip.model.utils import (
    _COMPILE_MODE_OPTIONS,
    _EAGER_MODEL_KEY,
    _TRAIN_TIME_SCRIPT_KEY,
)
from nequip.utils.logger import RankedLogger
from nequip.utils.versions import get_current_code_versions, _TORCH_GE_2_6
from nequip.utils.global_state import set_global_state

from ..__init__ import _DISCOVERED_NEQUIP_EXTENSION
from ._workflow_utils import set_workflow_state

import os
from omegaconf import OmegaConf
import hydra

# === setup logging ===
hydra.core.utils.configure_log(None)
logger = RankedLogger(__name__, rank_zero_only=True)


# `nequip-package` generates the archival format for NequIP framework models. This file contains the information necessary to track the archival format itself.
# whenever the archival format changes, `_CURRENT_NEQUIP_PACKAGE_VERSION` (counter to track the packaged model format) should be bumped up to the next number. We can then condition `ModelFromPackage` on the packaging format version to decide code paths to load the model appropriately.
# `nequip-package` format version index to condition other features upon when loading `nequip-package` from a specific version
_CURRENT_NEQUIP_PACKAGE_VERSION = 0


def main(args=None):

    parser = argparse.ArgumentParser(description="Package NequIP ecosystem models.")

    parser.add_argument(
        "--ckpt-path",
        help="path to checkpoint file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-path",
        help="output path to save the packaged model. NOTE: a `.nequip.zip` extension is mandatory",
        type=pathlib.Path,
        default=os.getcwd() + "/packaged_model.nequip.zip",
    )
    parser.add_argument(
        "--extra-externs",
        help="additional external modules to support during packaging",
        nargs="+",
        type=str,
        default=[],
    )

    args = parser.parse_args(args=args)

    set_workflow_state("package")

    assert str(args.output_path).endswith(
        ".nequip.zip"
    ), "`output-path` must end with the `.nequip.zip` extension"

    # === handle internal and external modules ===
    # internal and external modules that we know of
    _INTERNAL_MODULES = ["e3nn", "nequip"] + [
        ep.value for ep in _DISCOVERED_NEQUIP_EXTENSION
    ]
    # TODO: ideally we don't have any numpy or matplotlib dependencies, but for now it's here because of e3nn TPs
    _EXTERNAL_MODULES = ["triton", "io", "opt_einsum_fx", "numpy"] + args.extra_externs

    _MOCK_MODULES = ["matplotlib"]

    overlap = set(_INTERNAL_MODULES) & set(_EXTERNAL_MODULES)
    assert (
        not overlap
    ), f"Internal and external modules overlap with the following module(s): {overlap}"

    logger.debug("Internal Modules: " + str(_INTERNAL_MODULES))
    logger.debug("External Modules: " + str(_EXTERNAL_MODULES))

    # === load checkpoint and extract info ===
    checkpoint = torch.load(
        args.ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    # === text files to save ===
    # == original config ==
    orig_config = checkpoint["hyper_parameters"]["info_dict"].copy()
    # remove version info
    orig_config.pop("versions")
    orig_config = OmegaConf.to_yaml(orig_config)

    # == version info ==
    version_info = yaml.dump(
        get_current_code_versions(),
        default_flow_style=False,
    )

    # == set global state ==
    set_global_state()

    # == get example data from checkpoint ==
    # the reason for including it here is that whoever receives the packaged model file does not need to have access to the original data source to do `nequip-compile` on the packaged model (AOT export requires example data)
    logger.info("Instantiating datamodule for packaging.")
    data = data_dict_from_checkpoint(args.ckpt_path)

    # === perform packaging ===

    # the main complication is that we need to account for the possibility that the checkpoint is based on another packaged model
    # we then need to include the importer in the exporter for repackaging
    # see https://pytorch.org/docs/stable/package.html#re-export-an-imported-object
    # the reason for the present solution is due to the following constraints
    # 1. we need every model (eager, compile, aotinductor) to come from the same importer, which is used for constructing the exporter (we won't be able to save models coming from different importers, even if we pass all the importers during exporter construction)
    # 2. we need to use `ModelFromPackage` in a manner consistent with the API of fresh model building or `ModelFromCheckpoint` to ensure correct state restoration when loading the model from checkpoint (note that `nequip-package` is always loading a model from checkpoint with `ModelFromCheckpoint`)

    # the present solution is to first load an eager model from checkpoint,
    # which tells us whether the model originates from a package or not
    # if it's not from a package, we just instantiate all model build types and package
    # if it's from a package, we query a dict populated during `ModelFromPackage` (called by `ModelFromCheckpoints`) to get the `Importer` and the models [this ensures that all models come from the same `Importer`]
    # we also have to copy the state dict of the original eager model loaded (which should have its state restored correctly by the checkpoint load) into the other models in the dict

    # == get eager model ==
    # if the origin is `ModelFromPackage`, this call would have populated a dict with the relevant information to repackage
    logger.info(f"Building `{_EAGER_MODEL_KEY}` model for packaging ...")
    eager_model = ModelFromCheckpoint(args.ckpt_path, compile_mode=_EAGER_MODEL_KEY)

    if _TORCH_GE_2_6:
        # exclude train-time torchscript model (and eager since we've already loaded it)
        package_compile_modes = {
            mode
            for mode in _COMPILE_MODE_OPTIONS
            if mode not in [_EAGER_MODEL_KEY, _TRAIN_TIME_SCRIPT_KEY]
        }
    else:
        # only allow eager model if not torch>=2.6
        package_compile_modes = [_EAGER_MODEL_KEY]

    # == load models ==
    importers = (torch.package.importer.sys_importer,)
    models_to_package = {_EAGER_MODEL_KEY: eager_model}
    from nequip.model.from_save import _get_packaged_models

    pkg_models = _get_packaged_models()
    if not pkg_models:
        # the origin is not `ModelFromPackage`, so we just load the models one-by-one
        for compile_mode in package_compile_modes:
            logger.info(f"Building `{compile_mode}` model for packaging ...")
            model = ModelFromCheckpoint(args.ckpt_path, compile_mode=compile_mode)
            models_to_package.update({compile_mode: model})
    else:
        # the origin is `ModelFromPackage`
        # first update the `importers`
        importers = (pkg_models.pop("importer"),) + importers
        # then get the remaining models from the `pkg_models` dict and load the eager model's state dict into the other models
        # note that we use `pkg_models.keys()` instead of `package_compile_modes`
        # because the package may not have everything in `package_compile_modes`
        # e.g. if the original package was made with torch<2.6, and we're doing the current packaging with torch>=2.6
        pkg_modes = list(pkg_models.keys())
        for compile_mode in pkg_modes:
            logger.info(f"Building `{compile_mode}` model for packaging ...")
            model = pkg_models.pop(compile_mode)
            model.load_state_dict(eager_model.state_dict())
            models_to_package.update({compile_mode: model})

    # == package ==
    with warnings.catch_warnings():
        # suppress torch.package TypedStorage warning
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated.*",
            category=UserWarning,
            module="torch.package.package_exporter",
        )

        with torch.package.PackageExporter(
            args.output_path, importer=importers, debug=True
        ) as exp:
            # handle dependencies
            exp.mock([f"{pkg}.**" for pkg in _MOCK_MODULES])
            exp.extern([f"{pkg}.**" for pkg in _EXTERNAL_MODULES])
            exp.intern([f"{pkg}.**" for pkg in _INTERNAL_MODULES])
            # add data for aotinductor compilation
            exp.save_pickle(
                package="model",
                resource="example_data.pkl",
                obj=data,
                dependencies=True,
            )
            # save misc info
            exp.save_text("model", "config.yaml", orig_config)
            exp.save_text("model", "versions.txt", version_info)

            # save metadata used for loading packages
            pkg_metadata = {
                "package_version_id": _CURRENT_NEQUIP_PACKAGE_VERSION,
                "available_models": list(models_to_package.keys()),
            }
            exp.save_pickle(
                package="model",
                resource="package_metadata.pkl",
                obj=pkg_metadata,
            )

            # save models
            for compile_mode, model in models_to_package.items():
                exp.save_pickle(
                    package="model",
                    resource=f"{compile_mode}_model.pkl",
                    obj=model,
                    dependencies=True,
                )

            del importers

    logger.info(f"Packaged model saved to {args.output_path}")
    set_workflow_state(None)
    return


if __name__ == "__main__":
    main()
