import sys
import argparse
import textwrap
from pathlib import Path
import contextlib
from tqdm.auto import tqdm

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

            Prints only the final result in `name = num` format to stdout; all other information is printed to stderr.

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
        help="A YAML config file specifying the metrics to compute. If omitted, `config_final.yaml` in `train_dir` will be used. If the config does not specify `metrics_components`, the default is to print MAEs and RMSEs for all fields given in the loss function.",
        type=Path,
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
    # Something has to be provided
    # See https://stackoverflow.com/questions/22368458/how-to-make-argparse-print-usage-when-no-option-is-given-to-the-code
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
    # validate
    if args.dataset_config is None:
        raise ValueError("--dataset-config or --train-dir must be provided")
    if args.metrics_config is None:
        raise ValueError("--metrics-config or --train-dir must be provided")
    if args.model is None:
        raise ValueError("--model or --train-dir must be provided")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}", file=sys.stderr)
    if device.type == "cuda":
        print(
            "WARNING: please note that models running on CUDA are usually nondeterministc and that this manifests in the final test errors; for a _more_ deterministic result, please use `--device cpu`",
            file=sys.stderr,
        )

    # Load model:
    print("Loading model... ", file=sys.stderr, end="")
    try:
        model, _ = load_deployed_model(args.model, device=device)
        print("loaded deployed model.", file=sys.stderr)
    except ValueError:  # its not a deployed model
        model = torch.load(args.model, map_location=device)
        model = model.to(device)
        print("loaded pickled Python model.", file=sys.stderr)

    # Load a config file
    print(
        f"Loading {'original training ' if dataset_is_from_training else ''}dataset...",
        file=sys.stderr,
    )
    config = Config.from_file(str(args.dataset_config))

    # Currently, pytorch_geometric prints some status messages to stdout while loading the dataset
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
        print(
            f"Using training dataset minus training and validation frames, yielding a test set size of {len(test_idcs)} frames.",
            file=sys.stderr,
        )
    else:
        # load from file
        test_idcs = load_file(
            supported_formats=dict(
                torch=["pt", "pth"], yaml=["yaml", "yml"], json=["json"]
            ),
            filename=str(args.test_indexes),
        )
        print(
            f"Using provided test set indexes, yielding a test set size of {len(test_idcs)} frames.",
            file=sys.stderr,
        )

    # Figure out what metrics we're actually computing
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

    def _format_err(err: torch.Tensor, specifier: str):
        specifier = "{:" + specifier + "}"
        if err.nelement() == 1:
            return specifier.format(err.cpu().item())
        elif err.nelement() == 3:
            return (f"(x={specifier}, y={specifier}, z={specifier})").format(
                *err.cpu().squeeze().tolist()
            )
        else:
            raise AssertionError(
                "Somehow this metric configuration is unsupported, please file an issue!"
            )

    print("Starting...", file=sys.stderr)
    with tqdm(bar_format="{desc}") as display_bar:
        with tqdm(total=len(test_idcs)) as prog:
            while True:
                datas = [
                    dataset.get(int(idex))
                    for idex in test_idcs[
                        batch_i * batch_size : (batch_i + 1) * batch_size
                    ]
                ]
                if len(datas) == 0:
                    break
                batch = c.collate(datas)
                batch = batch.to(device)
                out = model(AtomicData.to_AtomicDataDict(batch))
                # Accumulate metrics
                with torch.no_grad():
                    metrics(out, batch)

                batch_i += 1
                display_bar.set_description_str(
                    " | ".join(
                        f"{k[0]}_{k[1]} = {_format_err(v, '4.2f')}"
                        for k, v in metrics.current_result().items()
                    )
                )
                prog.update(batch.num_graphs)
            display_bar.close()
        prog.close()

    print(file=sys.stderr)
    print("--- Final result: ---", file=sys.stderr)
    print(
        "\n".join(
            f"{k[0] + '_' + k[1]:>20s} = {_format_err(v, 'f'):<20s}"
            for k, v in metrics.current_result().items()
        )
    )


if __name__ == "__main__":
    main()
