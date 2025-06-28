# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import torch

from nequip.model.saved_models.checkpoint import data_dict_from_checkpoint
from nequip.model.saved_models.package import (
    _get_shared_importer,
    _get_package_metadata,
    _suppress_package_importer_warnings,
)
from nequip.model.saved_models import ModelFromCheckpoint
from nequip.model.utils import (
    _COMPILE_MODE_OPTIONS,
    _EAGER_MODEL_KEY,
)
from nequip.nn.model_modifier_utils import is_persistent_model_modifier
from nequip.model.modify_utils import get_all_modifiers, only_apply_persistent_modifiers
from nequip.utils.logger import RankedLogger
from nequip.utils.versions import get_current_code_versions, _TORCH_GE_2_6
from nequip.utils.versions.version_utils import get_version_safe
from nequip.utils.global_state import set_global_state

from ._workflow_utils import set_workflow_state
from ._package_utils import (
    _EXTERNAL_MODULES,
    _MOCK_MODULES,
    _INTERNAL_MODULES,
)

from omegaconf import OmegaConf
import hydra
import argparse
import pathlib
import yaml


# === setup logging ===
hydra.core.utils.configure_log(None)
logger = RankedLogger(__name__, rank_zero_only=True)


# `nequip-package` generates the archival format for NequIP framework models. This file contains the information necessary to track the archival format itself.
# whenever the archival format changes, `_CURRENT_NEQUIP_PACKAGE_VERSION` (counter to track the packaged model format) should be bumped up to the next number. We can then condition `ModelFromPackage` on the packaging format version to decide code paths to load the model appropriately.
# `nequip-package` format version index to condition other features upon when loading `nequip-package` from a specific version
#
# Package version high-level CHANGELOG:
# (use git blame on this line to identify specific commits and details of changes)
# 0:
#   - Initial version
# 1:
#   - package_metadata.txt instead of package_metadata.pkl
# 2:
#   - added `external_modules`
_CURRENT_NEQUIP_PACKAGE_VERSION = 2


def main(args=None):

    parser = argparse.ArgumentParser(description="Package NequIP ecosystem models.")

    subparsers = parser.add_subparsers(dest="command", title="commands")

    build_parser = subparsers.add_parser("build", help="build a packaged model file")
    build_parser.add_argument(
        "ckpt_path",
        help="path to checkpoint file",
        type=str,
    )
    build_parser.add_argument(
        "output_path",
        help="output path to save the packaged model. NOTE: a `.nequip.zip` extension is mandatory",
        type=pathlib.Path,
    )

    info_parser = subparsers.add_parser(
        "info", help="get information from a packaged model file"
    )
    # positional argument
    info_parser.add_argument(
        "pkg_path",
        help="path to package file",
        type=str,
    )
    info_parser.add_argument(
        "--get-modifiers",
        help="print all available model modifiers in the package file",
        action="store_true",
    )
    info_parser.add_argument(
        "--yaml",
        help="output in YAML format",
        action="store_true",
    )

    args = parser.parse_args(args=args)

    if args.command == "info":
        assert str(args.pkg_path).endswith(
            ".nequip.zip"
        ), "packed model file to inspect must end with the `.nequip.zip` extension"

        with _suppress_package_importer_warnings():
            imp = torch.package.PackageImporter(args.pkg_path)
            pkg_metadata = _get_package_metadata(imp)

            # Load and process modifiers if requested
            modifiers_info = None
            if args.get_modifiers:
                model = imp.load_pickle(
                    package="model",
                    resource=f"{_EAGER_MODEL_KEY}_model.pkl",
                    map_location="cpu",
                )
                modifiers_info = [
                    {
                        "name": name,
                        "persistent": is_persistent_model_modifier(modifier),
                        "doc": (
                            modifier.__doc__ if modifier.__doc__ is not None else ""
                        ).strip(),
                    }
                    for name, modifier in get_all_modifiers(model).items()
                ]

            # Print output
            if args.yaml:
                output_data = {"package_metadata": pkg_metadata}
                if modifiers_info is not None:
                    output_data["modifiers"] = modifiers_info
                print(yaml.dump(output_data))
            else:
                print("Package Metadata")
                print("================")
                print(
                    yaml.dump(
                        pkg_metadata,
                        default_flow_style=False,
                    )
                )
                if modifiers_info is not None:
                    print("Available Modifiers")
                    print("===================")
                    for idx, modifier_info in enumerate(modifiers_info):
                        persistent_flag = "" if modifier_info["persistent"] else "non-"
                        print(
                            f"{idx + 1}. {modifier_info['name']}\t({persistent_flag}persistent)\n"
                        )
                        if "doc" in modifier_info:
                            print(f"\t{modifier_info['doc']}\n")

        return

    elif args.command == "build":

        set_workflow_state("package")

        assert str(args.output_path).endswith(
            ".nequip.zip"
        ), "output path must end with the `.nequip.zip` extension"

        # === handle internal and external modules ===
        overlap = set(_INTERNAL_MODULES) & set(_EXTERNAL_MODULES)
        assert (
            not overlap
        ), f"Internal and external modules overlap with the following module(s): {overlap}"

        logger.debug("Internal Modules: " + str(_INTERNAL_MODULES))
        logger.debug("External Modules: " + str(_EXTERNAL_MODULES))
        logger.debug("Mock Modules: " + str(_MOCK_MODULES))

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
        code_versions = get_current_code_versions()

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
        # 1. we need every model to come from the same importer, which is used for constructing the exporter (we won't be able to save models coming from different importers, even if we pass all the importers during exporter construction)
        # 2. we need to use `ModelFromPackage` in a manner consistent with the API of fresh model building or `ModelFromCheckpoint` to ensure correct state restoration when loading the model from checkpoint (note that `nequip-package` is always loading a model from checkpoint with `ModelFromCheckpoint`)

        # the present solution is to first load an eager model from checkpoint,
        # which tells us whether the model originates from a package or not
        # if it's not from a package, we just instantiate all model build types and package
        # if it's from a package, we have logic in `ModelFromPackage` to reuse the same `Importer` as the one used to load the eager model [this ensures that all models come from the same `Importer`]
        # we use `ModelFromCheckpoint` to load the other models to make sure the relevant modifiers are applied and the states are restored correctly by the checkpoint load

        # == get eager model ==
        # if the origin is `ModelFromPackage`, this call would have populated a variable that we can query later
        logger.info(f"Building `{_EAGER_MODEL_KEY}` model for packaging ...")
        with only_apply_persistent_modifiers(persistent_only=True):
            eager_model = ModelFromCheckpoint(
                args.ckpt_path, compile_mode=_EAGER_MODEL_KEY
            )

        # it's a `ModuleDict`, so we just reach into one of the models to get `type_names`
        # we expect all models to have the same `type_names` (see init of base Lightning module in `nequip/train/lightning.py`)
        type_names = list(eager_model.values())[0].type_names

        # == load models ==
        importers = (torch.package.importer.sys_importer,)
        models_to_package = {_EAGER_MODEL_KEY: eager_model}

        # this variable is None if the origin is not a package
        imp = _get_shared_importer()
        if imp is not None:
            # the origin is `ModelFromPackage`
            # first update the `importers`
            importers = (imp,) + importers
            # we only repackage what's in the package
            # e.g. if the original package was made with torch<2.6, and we're doing the current packaging with torch>=2.6, we'll miss the `compile` model, but there's nothing we can do about it
            package_compile_modes = _get_package_metadata(imp)["available_models"]
        else:
            if _TORCH_GE_2_6:
                # allow everything (including compile models)
                package_compile_modes = _COMPILE_MODE_OPTIONS.copy()
            else:
                # only allow eager model if not torch>=2.6
                package_compile_modes = [_EAGER_MODEL_KEY]

        # remove eager model since we already built it
        package_compile_modes.remove(_EAGER_MODEL_KEY)

        for compile_mode in package_compile_modes:
            logger.info(f"Building `{compile_mode}` model for packaging ...")
            with only_apply_persistent_modifiers(persistent_only=True):
                model = ModelFromCheckpoint(args.ckpt_path, compile_mode=compile_mode)
            models_to_package.update({compile_mode: model})

        # == package ==
        with _suppress_package_importer_warnings():

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

                # save metadata used for loading packages
                pkg_metadata = {
                    "versions": code_versions,
                    "external_modules": {
                        k: get_version_safe(k) for k in _EXTERNAL_MODULES
                    },
                    "package_version_id": _CURRENT_NEQUIP_PACKAGE_VERSION,
                    "available_models": list(models_to_package.keys()),
                    "atom_types": {idx: name for idx, name in enumerate(type_names)},
                }
                pkg_metadata = yaml.dump(
                    pkg_metadata,
                    default_flow_style=False,
                )
                exp.save_text("model", "package_metadata.txt", pkg_metadata)

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
