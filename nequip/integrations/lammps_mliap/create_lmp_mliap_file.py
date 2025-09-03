# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import argparse
import pathlib

import torch

from .lmp_mliap_wrapper import NequIPLAMMPSMLIAPWrapper
from nequip.train.lightning import _SOLE_MODEL_KEY
from nequip.utils.logger import RankedLogger
from nequip.model.saved_models.load_utils import _get_model_file_path


logger = RankedLogger(__name__, rank_zero_only=True)


def main(args=None):
    # === parse inputs ===
    parser = argparse.ArgumentParser(
        description="Create NequIP LAMMPS ML-IAP file from saved models.",
    )

    # positional arguments:
    parser.add_argument(
        "model_path",
        help="path to a checkpoint model or packaged model file",
        type=pathlib.Path,
    )

    parser.add_argument(
        "output_path",
        help="path to write NequIP LAMMPS ML-IAP interface file (must end with `.nequip.lmp.pt`)",
        type=pathlib.Path,
    )

    # optional named arguments:
    parser.add_argument(
        "--model",
        help=f"name of model to use (default: {_SOLE_MODEL_KEY}, meant to work for the conventional single model case)",
        type=str,
        default=_SOLE_MODEL_KEY,
    )

    parser.add_argument(
        "--modifiers",
        help="modifiers to apply to the model",
        nargs="+",
        type=str,
        default=[],
    )

    parser.add_argument(
        "--no-compile",
        help="disable torch.compile for the model (default: False, compile is enabled)",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--tf32",
        help="whether to use TF32 or not (default: False)",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args(args=args)

    # === validate paths ===
    if not str(args.output_path).endswith(".nequip.lmp.pt"):
        raise ValueError(
            f"Output path must end with `.nequip.lmp.pt`, got: {args.output_path}"
        )

    # === create and save ML-IAP module ===
    logger.info(f"Creating LAMMPS ML-IAP artefact from {args.model_path} ...")

    # use `_get_model_file_path` to handle both local and nequip.net models
    with _get_model_file_path(str(args.model_path)) as model_file_path:
        mliap_module = NequIPLAMMPSMLIAPWrapper(
            model_path=str(model_file_path),
            model_key=args.model,
            modifiers=args.modifiers,
            compile=not args.no_compile,
            tf32=args.tf32,
        )
    torch.save(mliap_module, args.output_path)
    logger.info(f"LAMMPS ML-IAP artefact saved to {args.output_path}")


if __name__ == "__main__":
    main()
