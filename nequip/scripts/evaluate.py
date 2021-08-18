import sys
import argparse
from logging import (
    getLogger,
    CRITICAL,
    INFO,
    critical,
    info,
    StreamHandler,
    FileHandler,
)
import textwrap
from pathlib import Path
import contextlib
from tqdm.auto import tqdm

import ase.io

import torch

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicData, Collater
from nequip.scripts.deploy import load_deployed_model
from nequip.utils import load_file, instantiate
from nequip.train.loss import Loss
from nequip.train.metrics import Metrics


def main(args=None):
    # in results dir, do: nequip-deploy build . deployed.pth
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Compute the error of a model on a test set using various metrics.

            The model, metrics, dataset, etc. can specified individually, or a training session can be indicated with `--train-dir`.

            Prints only the final result in `name = num` format to stdout; all other information is logging.debuged to stderr.

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
        help="A YAML config file specifying the dataset to load test data from. If omitted, `config_final.yaml` in `train_dir` will be used",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--metrics-config",
        help="A YAML config file specifying the metrics to compute. If omitted, `config_final.yaml` in `train_dir` will be used. If the config does not specify `metrics_components`, the default is to logging.debug MAEs and RMSEs for all fields given in the loss function. If the literal string `None`, no metrics will be computed.",
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
        help="Batch size to use. Larger is usually faster on GPU.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--device",
        help="Device to run the model on. If not provided, defaults to CUDA if available and CPU otherwise.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output",
        help="XYZ file to write out the test set and model predicted forces, energies, etc. to.",
        type=Path,
        default=None,
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
            args.dataset_config = args.train_dir / "config_final.yaml"
            dataset_is_from_training = True
        if args.metrics_config is None:
            args.metrics_config = args.train_dir / "config_final.yaml"
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
    if args.output is not None:
        if args.output.suffix != ".xyz":
            raise ValueError("Only extxyz format for `--output` is supported.")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger = getLogger("")
    logger.setLevel(CRITICAL)
    logger.handlers = [StreamHandler(sys.stderr), StreamHandler(sys.stdout)]
    logger.handlers[0].setLevel(INFO)
    logger.handlers[1].setLevel(CRITICAL)
    if args.log is not None:
        logger.addHandler(FileHandler(args.log, mode="w"))
        logger.handlers[-1].setLevel(INFO)

    info(f"Using device: {device}")
    if device.type == "cuda":
        info(
            "WARNING: please note that models running on CUDA are usually nondeterministc and that this manifests in the final test errors; for a _more_ deterministic result, please use `--device cpu`",
        )

    # Load model:
    info("Loading model... ")
    try:
        model, _ = load_deployed_model(args.model, device=device)
        info("loaded deployed model.")
    except ValueError:  # its not a deployed model
        model = torch.load(args.model, map_location=device)
        model = model.to(device)
        info("loaded pickled Python model.")

    # Load a config file
    info(
        f"Loading {'original training ' if dataset_is_from_training else ''}dataset...",
    )
    config = Config.from_file(str(args.dataset_config))

    # Currently, pytorch_geometric debugs some status messages to stdout while loading the dataset
    # TODO: fix may come soon: https://github.com/rusty1s/pytorch_geometric/pull/2950
    # Until it does, just redirect them.
    with contextlib.redirect_stdout(sys.stderr):
        dataset = dataset_from_config(config)

    c = Collater.for_dataset(dataset, exclude_keys=[])

    # Determine the test set
    # this makes no sense if a dataset is given seperately
    if train_idcs is not None and dataset_is_from_training:
        # we know the train and val, get the rest
        all_idcs = set(range(len(dataset)))
        # set operations
        test_idcs = list(all_idcs - train_idcs - val_idcs)
        assert set(test_idcs).isdisjoint(train_idcs)
        assert set(test_idcs).isdisjoint(val_idcs)
        info(
            f"Using training dataset minus training and validation frames, yielding a test set size of {len(test_idcs)} frames.",
        )
        if not do_metrics:
            info(
                "WARNING: using the automatic test set ^^^ but not computing metrics, is this really what you wanted to do?",
            )
    else:
        # load from file
        test_idcs = load_file(
            supported_formats=dict(
                torch=["pt", "pth"], yaml=["yaml", "yml"], json=["json"]
            ),
            filename=str(args.test_indexes),
        )
        info(
            f"Using provided test set indexes, yielding a test set size of {len(test_idcs)} frames.",
        )

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

    info("Starting...")
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

        if args.output is not None:
            output = context_stack.enter_context(open(args.output, "w"))
        else:
            output = None

        while True:
            datas = [
                dataset.get(int(idex))
                for idex in test_idcs[batch_i * batch_size : (batch_i + 1) * batch_size]
            ]
            if len(datas) == 0:
                break
            batch = c.collate(datas)
            batch = batch.to(device)
            out = model(AtomicData.to_AtomicDataDict(batch))

            with torch.no_grad():
                # Write output
                if output is not None:
                    ase.io.write(
                        output,
                        AtomicData.from_AtomicDataDict(out).to(device="cpu").to_ase(),
                        format="extxyz",
                        append=True,
                    )
                # Accumulate metrics
                if do_metrics:
                    metrics(out, batch)
                    display_bar.set_description_str(
                        " | ".join(
                            f"{k} = {v:4.2f}"
                            for k, v in metrics.flatten_metrics(
                                metrics.current_result()
                            )[0].items()
                        )
                    )

            batch_i += 1
            prog.update(batch.num_graphs)

        prog.close()
        if do_metrics:
            display_bar.close()

    if do_metrics:
        info("\n--- Final result: ---")
        critical(
            "\n".join(
                f"{k:>20s} = {v:< 20f}"
                for k, v in metrics.flatten_metrics(metrics.current_result())[0].items()
            )
        )


if __name__ == "__main__":
    main()
