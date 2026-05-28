# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import torch

from nequip.model.saved_models.package import (
    _get_shared_importer,
    _get_package_metadata,
    _suppress_package_importer_exporter_warnings,
    _cpu_deserialize_if_no_cuda,
)
from nequip.model.saved_models import load_saved_model
from nequip.model.utils import _EAGER_MODEL_KEY
from nequip.nn.model_modifier_utils import (
    is_persistent_model_modifier,
    is_private_model_modifier,
)
from nequip.model.modify_utils import get_all_modifiers, modify
from nequip.utils.logger import RankedLogger
from nequip.utils.versions import get_current_code_versions
from nequip.utils.versions.version_utils import get_version_safe
from nequip.utils.global_state import set_global_state
from nequip.utils.asserts import assert_package_extension
from nequip.utils.dtype import test_model_output_similarity_by_dtype
from nequip.train.lightning import _SOLE_MODEL_KEY

from ._workflow_utils import set_workflow_state
from ._package_utils import (
    _EXTERNAL_MODULES,
    _MOCK_MODULES,
    _INTERNAL_MODULES,
    _PACKAGING_MODES,
)

from omegaconf import OmegaConf
import hydra
import argparse
import pathlib
import yaml
import zipfile
import difflib
import importlib.util
import shutil
import sys
import tempfile


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
    build_parser.add_argument(
        "--mode",
        help="packaging mode controlling which compile modes are included (default: nequip)",
        type=str,
        default="nequip",
    )

    subparsers.add_parser("modes", help="list available packaging modes")

    list_parser = subparsers.add_parser(
        "list", help="list files inside a packaged model file"
    )
    list_parser.add_argument(
        "pkg_path",
        help="path to package file",
        type=str,
    )

    diff_parser = subparsers.add_parser(
        "diff",
        help="diff a file inside a packaged model against an installed or local file",
    )
    diff_parser.add_argument("pkg_path", type=str, help="path to package file")
    diff_parser.add_argument(
        "zip_path",
        type=str,
        help="path of the file inside the zip (as shown by `nequip-package list`)",
    )
    diff_parser.add_argument(
        "local_file",
        type=str,
        nargs="?",
        default=None,
        help="local file to compare against; if omitted, auto-resolved from the installed package",
    )

    update_parser = subparsers.add_parser(
        "update",
        help="replace files in a packaged model and verify predictions are unchanged",
    )
    update_parser.add_argument(
        "input_path", type=pathlib.Path, help="path to input package file"
    )
    update_parser.add_argument(
        "output_path", type=pathlib.Path, help="path to output package file"
    )
    update_parser.add_argument(
        "--replace",
        action="append",
        default=[],
        nargs="+",
        metavar="ZIP_PATH",
        help="replace a file inside the zip; optionally follow with LOCAL_FILE, otherwise auto-resolved from the installed package. May be specified multiple times.",
    )

    modify_parser = subparsers.add_parser(
        "modify",
        help="apply a persistent model modifier to a packaged model",
    )
    modify_parser.add_argument(
        "input_path", type=pathlib.Path, help="path to input package file"
    )
    modify_parser.add_argument(
        "output_path", type=pathlib.Path, help="path to output package file"
    )
    modify_parser.add_argument(
        "--modifier",
        action="append",
        default=[],
        nargs="+",
        metavar="MODIFIER_NAME",
        help="modifier to apply; optionally follow with key=value kwargs (values are YAML-parsed). May be specified multiple times.",
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

    def _resolve_local_path(zip_path: str) -> pathlib.Path:
        """Resolve a zip-internal path to its installed-package counterpart.

        zip paths look like "<pkg_dir>/<top_level_module>/a/b/c.py";
        strip <pkg_dir>, find <top_level_module> in the current env via importlib.
        """
        parts = pathlib.PurePosixPath(zip_path).parts
        if len(parts) < 3:
            parser.error(
                f"cannot auto-resolve `{zip_path}`; provide a local file path explicitly"
            )
        top_level = parts[1]
        spec = importlib.util.find_spec(top_level)
        if spec is None or not spec.submodule_search_locations:
            parser.error(
                f"cannot auto-resolve: `{top_level}` is not an installed package; "
                f"provide a local file path explicitly"
            )
        return pathlib.Path(spec.submodule_search_locations[0]).joinpath(*parts[2:])

    if args.command == "modes":
        print("Available packaging modes:")
        for name, fn in _PACKAGING_MODES.items():
            print(f"  {name}\t{fn()}")
        return

    elif args.command == "list":
        assert_package_extension(args.pkg_path)
        with zipfile.ZipFile(args.pkg_path, "r") as zf:
            for info in sorted(zf.infolist(), key=lambda x: x.filename):
                if not info.filename.endswith(".storage"):
                    print(f"{info.file_size:>12}  {info.filename}")
        return

    elif args.command == "diff":
        assert_package_extension(args.pkg_path)

        # read file from the zip archive
        with zipfile.ZipFile(args.pkg_path, "r") as zf:
            pkg_bytes = zf.read(args.zip_path)

        # resolve the local counterpart to compare against
        if args.local_file is not None:
            local_path = pathlib.Path(args.local_file)
        else:
            local_path = _resolve_local_path(args.zip_path)

        if not local_path.exists():
            parser.error(f"local file not found: {local_path}")

        diff = list(
            difflib.unified_diff(
                pkg_bytes.decode("utf-8").splitlines(keepends=True),
                local_path.read_text(encoding="utf-8").splitlines(keepends=True),
                fromfile=f"{args.zip_path} (package)",
                tofile=str(local_path),
            )
        )
        # colorize when writing to a terminal: green=added, red=removed, cyan=hunk headers
        if diff and sys.stdout.isatty():
            _RESET, _RED, _GREEN, _CYAN = "\033[0m", "\033[31m", "\033[32m", "\033[36m"
            diff = [
                (
                    _GREEN
                    if l.startswith("+")
                    else _RED
                    if l.startswith("-")
                    else _CYAN
                    if l.startswith("@")
                    else ""
                )
                + l
                + (_RESET if l[0] in "+-@" else "")
                for l in diff
            ]
        sys.stdout.writelines(diff if diff else ["(no differences)\n"])

        return

    elif args.command == "update":
        assert_package_extension(args.input_path, "input package path")
        assert_package_extension(args.output_path, "output package path")
        assert args.input_path.resolve() != args.output_path.resolve(), (
            "input and output paths must differ"
        )

        # parse --replace ZIP_PATH [LOCAL_FILE] pairs
        # .py files → surgical source replacement via save_source_string
        # other files → text resource substitution (e.g. model/config.yaml)
        py_replacements = {}  # {module_name: (src_str, is_package)}
        text_replacements = {}  # {(package, resource): pathlib.Path}
        for item in args.replace:
            if len(item) == 1:
                zip_path = item[0]
                local_path = _resolve_local_path(zip_path)
            elif len(item) == 2:
                zip_path, local_file = item
                local_path = pathlib.Path(local_file)
            else:
                parser.error(
                    f"--replace takes 1 or 2 arguments, got {len(item)}: {item}"
                )
            parts = pathlib.PurePosixPath(zip_path).parts
            if len(parts) < 3:
                parser.error(
                    f"cannot parse zip path `{zip_path}`; expected <pkg_dir>/<package>/<resource>"
                )
            rel_path = "/".join(parts[1:])  # strip <pkg_dir>
            if rel_path.endswith(".py"):
                content = local_path.read_text(encoding="utf-8")
                if rel_path.endswith("/__init__.py"):
                    module_name = rel_path[: -len("/__init__.py")].replace("/", ".")
                    is_pkg = True
                else:
                    module_name = rel_path[:-3].replace("/", ".")
                    is_pkg = False
                py_replacements[module_name] = (content, is_pkg)
            else:
                pkg_name, resource = parts[1], "/".join(parts[2:])
                text_replacements[(pkg_name, resource)] = local_path

        # get reference predictions from the original package
        with _suppress_package_importer_exporter_warnings():
            old_imp = torch.package.PackageImporter(args.input_path)
            pkg_metadata = _get_package_metadata(old_imp)
            with _cpu_deserialize_if_no_cuda():
                old_example_data = old_imp.load_pickle("model", "example_data.pkl")
                old_module_dict = old_imp.load_pickle(
                    package="model",
                    resource=f"{_EAGER_MODEL_KEY}_model.pkl",
                    map_location="cpu",
                )
        ref_model = old_module_dict[_SOLE_MODEL_KEY]
        ref_model.eval()
        model_dtype = ref_model.model_dtype
        available_models = pkg_metadata["available_models"]

        # write to a tempfile so a failed verification never produces a bad output file
        with tempfile.NamedTemporaryFile(suffix=".nequip.zip", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            # re-package via PackageExporter:
            # - unchanged Python source comes from old_imp (first in importer chain)
            # - .py replacements are injected via save_source_string before intern resolution;
            #   save_pickle(dependencies=True) skips modules already in the package, so
            #   the injected source wins and intern leaves them alone
            with _suppress_package_importer_exporter_warnings():
                importers = (old_imp, torch.package.importer.sys_importer)
                with torch.package.PackageExporter(tmp_path, importer=importers) as exp:
                    exp.mock([f"{pkg}.**" for pkg in _MOCK_MODULES])
                    exp.extern([f"{pkg}.**" for pkg in _EXTERNAL_MODULES])
                    exp.intern([f"{pkg}.**" for pkg in _INTERNAL_MODULES])

                    # inject .py replacements before pickle saving so intern won't overwrite them
                    for module_name, (src, is_pkg) in py_replacements.items():
                        logger.info(
                            f"Replacing {module_name.replace('.', '/')}"
                            f"{'/__init__' if is_pkg else ''}.py"
                        )
                        exp.save_source_string(module_name, src, is_package=is_pkg)

                    # example data: always carry over from old package
                    exp.save_pickle(
                        "model", "example_data.pkl", old_example_data, dependencies=True
                    )

                    # text resources: carry over from old package, substituting replacements
                    unused_text = set(text_replacements.keys())
                    for resource in ["config.yaml", "package_metadata.txt"]:
                        key = ("model", resource)
                        if key in text_replacements:
                            content = text_replacements[key].read_text(encoding="utf-8")
                            logger.info(
                                f"Replacing model/{resource} from {text_replacements[key]}"
                            )
                            unused_text.discard(key)
                        else:
                            content = old_imp.load_text("model", resource)
                        exp.save_text("model", resource, content)

                    if unused_text:
                        logger.warning(
                            "The following --replace targets were not used "
                            "(expected model/config.yaml or model/package_metadata.txt): "
                            + ", ".join(f"{p}/{r}" for p, r in unused_text)
                        )

                    # model pickles: load from old package and re-save (weights preserved)
                    for compile_mode in available_models:
                        with _cpu_deserialize_if_no_cuda():
                            model_dict = old_imp.load_pickle(
                                package="model",
                                resource=f"{compile_mode}_model.pkl",
                                map_location="cpu",
                            )
                        exp.save_pickle(
                            "model",
                            f"{compile_mode}_model.pkl",
                            model_dict,
                            dependencies=True,
                        )

            # verify predictions are unchanged using the original example data
            with _suppress_package_importer_exporter_warnings():
                new_imp = torch.package.PackageImporter(tmp_path)
                with _cpu_deserialize_if_no_cuda():
                    new_module_dict = new_imp.load_pickle(
                        package="model",
                        resource=f"{_EAGER_MODEL_KEY}_model.pkl",
                        map_location="cpu",
                    )
            new_model = new_module_dict[_SOLE_MODEL_KEY]
            new_model.eval()

            def _error_msg(key, tol, err, absval, model_dtype):
                return (
                    f"package update verification failed for field `{key}`: "
                    f"MaxAbsError {err:.6g} exceeds tolerance {tol} ({model_dtype} model), "
                    f"MaxAbs value {absval:.6g}. "
                    f"Set NEQUIP_FLOAT32_MODEL_TOL / NEQUIP_FLOAT64_MODEL_TOL to adjust."
                )

            test_model_output_similarity_by_dtype(
                ref_model,
                new_model,
                old_example_data,
                model_dtype,
                error_message=_error_msg,
            )

            shutil.move(tmp_path, args.output_path)
        except:
            tmp_path.unlink(missing_ok=True)
            raise

        logger.info(
            f"Updated package saved to {args.output_path} (predictions verified)"
        )
        return

    elif args.command == "modify":
        assert_package_extension(args.input_path, "input package path")
        assert_package_extension(args.output_path, "output package path")
        assert args.input_path.resolve() != args.output_path.resolve(), (
            "input and output paths must differ"
        )

        # parse --modifier NAME [key=value ...] into modifier config dicts
        modifiers_config = []
        for item in args.modifier:
            name = item[0]
            kwargs = {}
            for kv in item[1:]:
                k, v = kv.split("=", 1)
                kwargs[k] = yaml.safe_load(v)
            modifiers_config.append({"modifier": name, **kwargs})

        with _suppress_package_importer_exporter_warnings():
            old_imp = torch.package.PackageImporter(args.input_path)
            pkg_metadata = _get_package_metadata(old_imp)
            with _cpu_deserialize_if_no_cuda():
                old_example_data = old_imp.load_pickle("model", "example_data.pkl")
                eager_dict = old_imp.load_pickle(
                    package="model",
                    resource=f"{_EAGER_MODEL_KEY}_model.pkl",
                    map_location="cpu",
                )

        # validate: modifiers must be registered, persistent, and public
        avail = get_all_modifiers(eager_dict[_SOLE_MODEL_KEY])
        for cfg in modifiers_config:
            name = cfg["modifier"]
            if name not in avail:
                persistent_public = [
                    n
                    for n, f in avail.items()
                    if is_persistent_model_modifier(f)
                    and not is_private_model_modifier(f)
                ]
                raise ValueError(
                    f"`{name}` is not a registered modifier. "
                    f"Available persistent public modifiers: {persistent_public}"
                )
            if not is_persistent_model_modifier(avail[name]):
                raise ValueError(
                    f"`{name}` is not a persistent modifier; `nequip-package modify` only applies persistent modifiers"
                )
            if is_private_model_modifier(avail[name]):
                raise ValueError(f"`{name}` is a private modifier")

        with tempfile.NamedTemporaryFile(suffix=".nequip.zip", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            with _suppress_package_importer_exporter_warnings():
                importers = (old_imp, torch.package.importer.sys_importer)
                with torch.package.PackageExporter(tmp_path, importer=importers) as exp:
                    exp.mock([f"{pkg}.**" for pkg in _MOCK_MODULES])
                    exp.extern([f"{pkg}.**" for pkg in _EXTERNAL_MODULES])
                    exp.intern([f"{pkg}.**" for pkg in _INTERNAL_MODULES])

                    exp.save_pickle(
                        "model", "example_data.pkl", old_example_data, dependencies=True
                    )
                    for resource in ["config.yaml", "package_metadata.txt"]:
                        exp.save_text(
                            "model", resource, old_imp.load_text("model", resource)
                        )

                    for compile_mode in pkg_metadata["available_models"]:
                        with _cpu_deserialize_if_no_cuda():
                            model_dict = old_imp.load_pickle(
                                package="model",
                                resource=f"{compile_mode}_model.pkl",
                                map_location="cpu",
                            )
                        modify(model_dict, modifiers_config)
                        exp.save_pickle(
                            "model",
                            f"{compile_mode}_model.pkl",
                            model_dict,
                            dependencies=True,
                        )

            shutil.move(tmp_path, args.output_path)
        except:
            tmp_path.unlink(missing_ok=True)
            raise

        logger.info(f"Modified package saved to {args.output_path}")
        return

    elif args.command == "info":
        assert_package_extension(args.pkg_path)

        with _suppress_package_importer_exporter_warnings():
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
                    if not is_private_model_modifier(modifier)
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

        assert_package_extension(args.output_path, "output package path")

        # === handle internal and external modules ===
        overlap = set(_INTERNAL_MODULES) & set(_EXTERNAL_MODULES)
        assert not overlap, (
            f"Internal and external modules overlap with the following module(s): {overlap}"
        )

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
        eager_model, data = load_saved_model(
            args.ckpt_path,
            compile_mode=_EAGER_MODEL_KEY,
            model_key=None,
            return_data_dict=True,
            # ^ get example data from checkpoint/package
            # the reason for including it here is that whoever receives the packaged model file does not need to have access to the original data source to do `nequip-compile` on the packaged model (AOT export requires example data)
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
            assert args.mode in _PACKAGING_MODES, (
                f"unknown packaging mode '{args.mode}'. "
                f"Available modes: {list(_PACKAGING_MODES.keys())}"
            )
            package_compile_modes = _PACKAGING_MODES[args.mode]()
            assert _EAGER_MODEL_KEY in package_compile_modes, (
                f"packaging mode '{args.mode}' must include '{_EAGER_MODEL_KEY}'"
            )

        # remove eager model since we already built it
        package_compile_modes.remove(_EAGER_MODEL_KEY)

        for compile_mode in package_compile_modes:
            logger.info(f"Building `{compile_mode}` model for packaging ...")
            model = load_saved_model(
                args.ckpt_path, compile_mode=compile_mode, model_key=None
            )
            models_to_package.update({compile_mode: model})

        # == package ==
        with _suppress_package_importer_exporter_warnings():
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
                    # make sure model is on CPU before packaging
                    model = model.to(torch.device("cpu"))
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
