from typing import Optional
import sys
import argparse
import logging
import textwrap
from pathlib import Path
import contextlib
from tqdm.auto import tqdm

import ase.io

import torch

from nequip.data import AtomicData, Collater, dataset_from_config, register_fields
from nequip.scripts.deploy import load_deployed_model, R_MAX_KEY
from nequip.scripts._logger import set_up_script_logger
from nequip.scripts.train import default_config, check_code_version
from nequip.utils._global_options import _set_global_options
from nequip.train import Trainer, Loss, Metrics
from nequip.utils import load_file, instantiate, Config


ORIGINAL_DATASET_INDEX_KEY: str = "original_dataset_index"
register_fields(graph_fields=[ORIGINAL_DATASET_INDEX_KEY])


def main(args=None, running_as_script: bool = True):
    # in results dir, do: nequip-deploy build --train-dir . deployed.pth
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Compute the error of a model on a test set using various metrics.

            The model, metrics, dataset, etc. can specified in individual YAML config files, or a training session can be indicated with `--train-dir`.
            In order of priority, the global settings (dtype, TensorFloat32, etc.) are taken from:
              (1) the model config (for a training session),
              (2) the dataset config (for a deployed model),
              or (3) the defaults.

            Prints only the final result in `name = num` format to stdout; all other information is `logging.debug`ed to stderr.

            WARNING: Please note that results of CUDA models are rarely exactly reproducible, and that even CPU models can be nondeterministic.
            """
        )
    )
    parser.add_argument(
        "--train-dir",
        help="Path to a working directory from a training session.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model",
        help="A deployed or pickled NequIP model to load. If omitted, defaults to `best_model.pth` in `train_dir`.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--dataset-config",
        help="A YAML config file specifying the dataset to load test data from. If omitted, `config.yaml` in `train_dir` will be used",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--metrics-config",
        help="A YAML config file specifying the metrics to compute. If omitted, `config.yaml` in `train_dir` will be used. If the config does not specify `metrics_components`, the default is to logging.debug MAEs and RMSEs for all fields given in the loss function. If the literal string `None`, no metrics will be computed.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test-indexes",
        help="Path to a file containing the indexes in the dataset that make up the test set. If omitted, all data frames *not* used as training or validation data in the training session `train_dir` will be used.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size to use. Larger is usually faster on GPU. If you run out of memory, lower this. You can also try to raise this for faster evaluation. Default: 50.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--repeat",
        help=(
            "Number of times to repeat evaluating the test dataset. "
            "This can help compensate for CUDA nondeterminism, or can be used to evaluate error on models whose inference passes are intentionally nondeterministic. "
            "Note that `--repeat`ed passes over the dataset will also be `--output`ed if an `--output` is specified."
        ),
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        help="Try to have PyTorch use deterministic algorithms. Will probably fail on GPU/CUDA.",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--device",
        help="Device to run the model on. If not provided, defaults to CUDA if available and CPU otherwise.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output",
        help="ExtXYZ (.xyz) file to write out the test set and model predictions to.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-fields",
        help="Extra fields (names comma separated with no spaces) to write to the `--output`.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--log",
        help="log file to store all the metrics and screen logging.debug",
        type=Path,
        default=None,
    )
    # Something has to be provided
    # See https://stackoverflow.com/questions/22368458/how-to-make-argparse-logging.debug-usage-when-no-option-is-given-to-the-code
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    # Parse the args
    args = parser.parse_args(args=args)

    # Do the defaults:
    dataset_is_from_training: bool = False
    if args.train_dir:
        if args.dataset_config is None:
            args.dataset_config = args.train_dir / "config.yaml"
            dataset_is_from_training = True
        if args.metrics_config is None:
            args.metrics_config = args.train_dir / "config.yaml"
        if args.model is None:
            args.model = args.train_dir / "best_model.pth"
        if args.test_indexes is None:
            # Find the remaining indexes that arent train or val
            trainer = torch.load(
                str(args.train_dir / "trainer.pth"), map_location="cpu"
            )
            train_idcs = set(trainer["train_idcs"].tolist())
            val_idcs = set(trainer["val_idcs"].tolist())
        else:
            train_idcs = val_idcs = None
    # update
    if args.metrics_config == "None":
        args.metrics_config = None
    elif args.metrics_config is not None:
        args.metrics_config = Path(args.metrics_config)
    do_metrics = args.metrics_config is not None
    # validate
    if args.dataset_config is None:
        raise ValueError("--dataset-config or --train-dir must be provided")
    if args.metrics_config is None and args.output is None:
        raise ValueError(
            "Nothing to do! Must provide at least one of --metrics-config, --train-dir (to use training config for metrics), or --output"
        )
    if args.model is None:
        raise ValueError("--model or --train-dir must be provided")
    output_type: Optional[str] = None
    if args.output is not None:
        if args.output.suffix != ".xyz":
            raise ValueError("Only .xyz format for `--output` is supported.")
        args.output_fields = [e for e in args.output_fields.split(",") if e != ""] + [
            ORIGINAL_DATASET_INDEX_KEY
        ]
        output_type = "xyz"
    else:
        assert args.output_fields == ""
        args.output_fields = []

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if running_as_script:
        set_up_script_logger(args.log)
    logger = logging.getLogger("nequip-evaluate")
    logger.setLevel(logging.INFO)

    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(
            "WARNING: please note that models running on CUDA are usually nondeterministc and that this manifests in the final test errors; for a _more_ deterministic result, please use `--device cpu`",
        )

    if args.use_deterministic_algorithms:
        logger.info(
            "Telling PyTorch to try to use deterministic algorithms... please note that this will likely error on CUDA/GPU"
        )
        torch.use_deterministic_algorithms(True)

    # Load model:
    logger.info("Loading model... ")
    loaded_deployed_model: bool = False
    model_r_max = None
    try:
        model, metadata = load_deployed_model(
            args.model,
            device=device,
            set_global_options=True,  # don't warn that setting
        )
        logger.info("loaded deployed model.")
        # the global settings for a deployed model are set by
        # set_global_options in the call to load_deployed_model
        # above
        model_r_max = float(metadata[R_MAX_KEY])
        loaded_deployed_model = True
    except ValueError:  # its not a deployed model
        loaded_deployed_model = False
    # we don't do this in the `except:` block to avoid "during handing of this exception another exception"
    # chains if there is an issue loading the training session model. This makes the error messages more
    # comprehensible:
    if not loaded_deployed_model:
        # Use the model config, regardless of dataset config
        global_config = args.model.parent / "config.yaml"
        global_config = Config.from_file(str(global_config), defaults=default_config)
        _set_global_options(global_config)
        check_code_version(global_config)
        del global_config

        # load a training session model
        model, model_config = Trainer.load_model_from_training_session(
            traindir=args.model.parent, model_name=args.model.name
        )
        model = model.to(device)
        logger.info("loaded model from training session")
        model_r_max = model_config["r_max"]
    model.eval()

    # Load a config file
    logger.info(
        f"Loading {'original ' if dataset_is_from_training else ''}dataset...",
    )
    dataset_config = Config.from_file(
        str(args.dataset_config), defaults={"r_max": model_r_max}
    )
    if dataset_config["r_max"] != model_r_max:
        raise RuntimeError(
            f"Dataset config has r_max={dataset_config['r_max']}, but model has r_max={model_r_max}!"
        )

    dataset_is_validation: bool = False
    try:
        # Try to get validation dataset
        dataset = dataset_from_config(dataset_config, prefix="validation_dataset")
        dataset_is_validation = True
    except KeyError:
        pass
    if not dataset_is_validation:
        # Get shared train + validation dataset
        # prefix `dataset`
        dataset = dataset_from_config(dataset_config)
    logger.info(
        f"Loaded {'validation_' if dataset_is_validation else ''}dataset specified in {args.dataset_config.name}.",
    )

    c = Collater.for_dataset(dataset, exclude_keys=[])

    # Determine the test set
    # this makes no sense if a dataset is given seperately
    if (
        args.test_indexes is None
        and dataset_is_from_training
        and train_idcs is not None
    ):
        # we know the train and val, get the rest
        all_idcs = set(range(len(dataset)))
        # set operations
        if dataset_is_validation:
            test_idcs = list(all_idcs - val_idcs)
            logger.info(
                f"Using origial validation dataset ({len(dataset)} frames) minus validation set frames ({len(val_idcs)} frames), yielding a test set size of {len(test_idcs)} frames.",
            )
        else:
            test_idcs = list(all_idcs - train_idcs - val_idcs)
            assert set(test_idcs).isdisjoint(train_idcs)
            logger.info(
                f"Using origial training dataset ({len(dataset)} frames) minus training ({len(train_idcs)} frames) and validation frames ({len(val_idcs)} frames), yielding a test set size of {len(test_idcs)} frames.",
            )
        # No matter what it should be disjoint from validation:
        assert set(test_idcs).isdisjoint(val_idcs)
        if not do_metrics:
            logger.info(
                "WARNING: using the automatic test set ^^^ but not computing metrics, is this really what you wanted to do?",
            )
    elif args.test_indexes is None:
        # Default to all frames
        test_idcs = torch.arange(dataset.len())
        logger.info(
            f"Using all frames from the specified test dataset, yielding a test set size of {len(test_idcs)} frames.",
        )
    else:
        # load from file
        test_idcs = load_file(
            supported_formats=dict(
                torch=["pt", "pth"], yaml=["yaml", "yml"], json=["json"]
            ),
            filename=str(args.test_indexes),
        )
        logger.info(
            f"Using provided test set indexes, yielding a test set size of {len(test_idcs)} frames.",
        )
    test_idcs = torch.as_tensor(test_idcs, dtype=torch.long)
    test_idcs = test_idcs.tile((args.repeat,))

    # Figure out what metrics we're actually computing
    if do_metrics:
        metrics_config = Config.from_file(str(args.metrics_config))
        metrics_components = metrics_config.get("metrics_components", None)
        # See trainer.py: init() and init_metrics()
        # Default to loss functions if no metrics specified:
        if metrics_components is None:
            loss, _ = instantiate(
                builder=Loss,
                prefix="loss",
                positional_args=dict(coeffs=metrics_config.loss_coeffs),
                all_args=metrics_config,
            )
            metrics_components = []
            for key, func in loss.funcs.items():
                params = {
                    "PerSpecies": type(func).__name__.startswith("PerSpecies"),
                }
                metrics_components.append((key, "mae", params))
                metrics_components.append((key, "rmse", params))

        metrics, _ = instantiate(
            builder=Metrics,
            prefix="metrics",
            positional_args=dict(components=metrics_components),
            all_args=metrics_config,
        )
        metrics.to(device=device)

    batch_i: int = 0
    batch_size: int = args.batch_size

    logger.info("Starting...")
    context_stack = contextlib.ExitStack()
    with contextlib.ExitStack() as context_stack:
        # "None" checks if in a TTY and disables if not
        prog = context_stack.enter_context(tqdm(total=len(test_idcs), disable=None))
        if do_metrics:
            display_bar = context_stack.enter_context(
                tqdm(
                    bar_format=""
                    if prog.disable  # prog.ncols doesn't exist if disabled
                    else ("{desc:." + str(prog.ncols) + "}"),
                    disable=None,
                )
            )

        if output_type is not None:
            output = context_stack.enter_context(open(args.output, "w"))
        else:
            output = None

        while True:
            this_batch_test_indexes = test_idcs[
                batch_i * batch_size : (batch_i + 1) * batch_size
            ]
            datas = [dataset[int(idex)] for idex in this_batch_test_indexes]
            if len(datas) == 0:
                break
            batch = c.collate(datas)
            batch = batch.to(device)
            out = model(AtomicData.to_AtomicDataDict(batch))

            with torch.no_grad():
                # Write output
                if output_type == "xyz":
                    # add test frame to the output:
                    out[ORIGINAL_DATASET_INDEX_KEY] = torch.LongTensor(
                        this_batch_test_indexes
                    )
                    # append to the file
                    ase.io.write(
                        output,
                        AtomicData.from_AtomicDataDict(out)
                        .to(device="cpu")
                        .to_ase(
                            type_mapper=dataset.type_mapper,
                            extra_fields=args.output_fields,
                        ),
                        format="extxyz",
                        append=True,
                    )

                # Accumulate metrics
                if do_metrics:
                    metrics(out, batch)
                    display_bar.set_description_str(
                        " | ".join(
                            f"{k} = {v:4.4f}"
                            for k, v in metrics.flatten_metrics(
                                metrics.current_result(),
                                type_names=dataset.type_mapper.type_names,
                            )[0].items()
                        )
                    )

            batch_i += 1
            prog.update(batch.num_graphs)

        prog.close()
        if do_metrics:
            display_bar.close()

    if do_metrics:
        logger.info("\n--- Final result: ---")
        logger.critical(
            "\n".join(
                f"{k:>20s} = {v:< 20f}"
                for k, v in metrics.flatten_metrics(
                    metrics.current_result(),
                    type_names=dataset.type_mapper.type_names,
                )[0].items()
            )
        )


if __name__ == "__main__":
    main(running_as_script=True)
