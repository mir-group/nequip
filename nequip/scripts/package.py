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
from nequip.model import ModelFromCheckpoint, override_model_compile_mode
from nequip.utils.logger import RankedLogger
from nequip.utils.versions import get_current_code_versions
from nequip.utils.global_state import set_global_state

from ..__init__ import _DISCOVERED_NEQUIP_EXTENSION

import os
from omegaconf import OmegaConf
import hydra

# === setup logging ===
hydra.core.utils.configure_log(None)
logger = RankedLogger(__name__, rank_zero_only=True)


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
    # TODO: enable overriding logic when a use case arises
    parser.add_argument(
        "--override-model",
        help="add or override model configuration keys from the checkpoint file -- unless you know why you need to, do not use this option.",
        type=str,
        default=None,
    )

    args = parser.parse_args(args=args)

    assert str(args.output_path).endswith(
        ".nequip.zip"
    ), "`output-path` must end with the `.nequip.zip` extension"

    # === handle internal and external modules ===
    # internal and external modules that we know of
    _INTERNAL_MODULES = ["nequip"] + [ep.value for ep in _DISCOVERED_NEQUIP_EXTENSION]
    # TODO: make e3nn intern eventually (requires refactoring e3nn code)
    _EXTERNAL_MODULES = ["e3nn", "opt_einsum_fx"] + args.extra_externs

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

    # == build model from checkpoint ==
    # pickle model without torchscript or torch.compile
    with override_model_compile_mode(compile_mode=None):
        model = ModelFromCheckpoint(args.ckpt_path)

    # == get example data from checkpoint ==
    # the reason for including it here is that whoever receives the packaged model file does not need to have access to the original data source to do `nequip-compile` on the packaged model (AOT export requires example data)
    logger.info("Instantiating datamodule for packaging.")
    data = data_dict_from_checkpoint(args.ckpt_path)

    with warnings.catch_warnings():
        # suppress torch.package TypedStorage warning
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated.*",
            category=UserWarning,
            module="torch.package.package_exporter",
        )

        with torch.package.PackageExporter(args.output_path, debug=True) as exp:
            exp.extern([f"{pkg}.**" for pkg in _EXTERNAL_MODULES])
            exp.intern([f"{pkg}.**" for pkg in _INTERNAL_MODULES])

            exp.save_pickle(
                package="model",
                resource="model.pkl",
                obj=model,
                dependencies=True,
            )
            exp.save_pickle(
                package="model",
                resource="example_data.pkl",
                obj=data,
                dependencies=True,
            )
            exp.save_text("model", "config.yaml", orig_config)
            exp.save_text("model", "versions.txt", version_info)
            exp.save_text(
                "model",
                "nequip_package_version_id.txt",
                str(_CURRENT_NEQUIP_PACKAGE_VERSION),
            )

    logger.info(f"Packaged model saved to {args.output_path}")

    return


if __name__ == "__main__":
    main()
