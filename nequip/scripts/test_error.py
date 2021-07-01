import sys
import argparse
from pathlib import Path

import torch

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, Collater
from nequip.scripts.deploy import load_deployed_model
from nequip.utils import load_file

from torch_runstats import RunningStats


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
        "--test-indexes",
        help="Path to a file containing the indexes in the dataset that make up the test set. If omitted, all data frames *not* used as training or validation data in the training session `train_dir` will be used.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size to use. Larger is usually faster on GPU.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--log-every",
        help="Log approximately every n datapoints.",
        type=int,
        default=10,
    )
    args = parser.parse_args(args=args)

    # Do the defaults:
    if args.train_dir:
        if args.dataset_config is None:
            args.dataset_config = args.train_dir / "config_final.yaml"
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
    if args.model is None:
        raise ValueError("--model or --train-dir must be provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model:
    try:
        model, _ = load_deployed_model(args.model, device=device)
    except ValueError:  # its not a deployed model
        model = torch.load(args.model, map_location=device)

    # Load a config file
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

    # Do the stats
    e_stats = RunningStats()
    e_stats.to(device=device, dtype=torch.get_default_dtype())
    f_stats = RunningStats(dim=(3,), reduce_dims=(0,))
    f_stats.to(device=device, dtype=torch.get_default_dtype())

    batch_i: int = 0
    batch_size: int = args.batch_size
    since_last_log: int = 0

    while True:
        datas = [
            dataset.get(int(idex))
            for idex in test_idcs[batch_i * batch_size : (batch_i + 1) * batch_size]
        ]
        since_last_log += len(datas)
        if len(datas) == 0:
            break
        batch = c.collate(datas)
        batch = batch.to(device)
        out = model(AtomicData.to_AtomicDataDict(batch))
        e = out[AtomicDataDict.TOTAL_ENERGY_KEY].detach()
        f = out[AtomicDataDict.FORCE_KEY].detach()
        e_stats.accumulate_batch((e - batch[AtomicDataDict.TOTAL_ENERGY_KEY]).abs())
        f_stats.accumulate_batch((f - batch[AtomicDataDict.FORCE_KEY]).abs())

        if since_last_log >= args.log_every:
            print(
                "Progress: {:.2f}%, cumulative MAE-F: {}, cumulative MAE-E: {}".format(
                    (e_stats.n.cpu().item() * 100) / len(test_idcs),
                    e_stats.current_result().cpu().item(),
                    f_stats.current_result().cpu().item(),
                ),
                file=sys.stderr,
            )
            since_last_log = 0

        batch_i += 1

    print("Force MAE: {}".format(f_stats.current_result().cpu().item()))
    print("Energy MAE: {}".format(e_stats.current_result().cpu().item()))


if __name__ == "__main__":
    main()
