import sys
import argparse
from pathlib import Path
from numpy import disp
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
    parser = argparse.ArgumentParser()
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
        help="Device to run the model on. If not provided, defaults to CUDA if available and CPU otherwise. Please note that results of CUDA models are rarely exactly reproducible, and that even CPU models can be nondeterministic.",
        type=str,
        default=None,
    )
    args = parser.parse_args(args=args)

    # Do the defaults:
    if args.train_dir:
        if args.dataset_config is None:
            args.dataset_config = args.train_dir / "config_final.yaml"
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
    print("Loading dataset...", file=sys.stderr)
    config = Config.from_file(str(args.dataset_config))
    dataset = dataset_from_config(config)
    c = Collater.for_dataset(dataset, exclude_keys=[])

    # Determine the test set
    if train_idcs is not None:
        # we know the train and val, get the rest
        all_idcs = set(range(len(dataset)))
        # set operations
        test_idcs = list(all_idcs - train_idcs - val_idcs)
        assert set(test_idcs).isdisjoint(train_idcs)
        assert set(test_idcs).isdisjoint(val_idcs)
    else:
        # load from file
        test_idcs = load_file(args.test_indexes)

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
                        f"{k[0]}_{k[1]} = {v.cpu().item(): 4.2f}"
                        for k, v in metrics.current_result().items()
                    )
                )
                prog.update(batch.num_graphs)
            display_bar.close()
        prog.close()

    print("--- Final result: ---")
    print(
        "\n".join(
            f"{k[0] + '_' + k[1]:>20s}  = {v.cpu().item():< 20f}"
            for k, v in metrics.current_result().items()
        )
    )


if __name__ == "__main__":
    main()
