import sys
import argparse
import logging
import textwrap
import contextlib
import itertools

import torch
from torch.utils.benchmark import Timer

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicData, AtomicDataDict
from nequip.model import model_from_config
from nequip.scripts.deploy import _compile_for_deploy
from nequip.scripts.train import _set_global_options, default_config


def main(args=None, running_as_script: bool = True):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Benchmark the approximate MD performance of a given model configuration / dataset pair."""
        )
    )
    parser.add_argument("config", help="configuration file")
    parser.add_argument(
        "--device",
        help="Device to run the model on. If not provided, defaults to CUDA if available and CPU otherwise.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-n", help="Number of trials.", type=int, default=30,
    )
    parser.add_argument(
        "--n-data", help="Number of frames to use.", type=int, default=1,
    )

    # TODO: option to profile
    # TODO: option to show memory use

    # Parse the args
    args = parser.parse_args(args=args)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if running_as_script:
        # Configure the root logger so stuff gets printed
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("nequip-evaluate")
    logger.setLevel(logging.INFO)

    logger.info(f"Using device: {device}")

    config = Config.from_file(args.config, defaults=default_config)
    _set_global_options(config)

    # Load dataset to get something to benchmark on
    logger.info("Loading dataset... ")
    # Currently, pytorch_geometric prints some status messages to stdout while loading the dataset
    # TODO: fix may come soon: https://github.com/rusty1s/pytorch_geometric/pull/2950
    # Until it does, just redirect them.
    with contextlib.redirect_stdout(sys.stderr):
        dataset = dataset_from_config(config)
    datas = itertools.cycle(
        [
            AtomicData.to_AtomicDataDict(dataset[i].to(device))
            for i in torch.randperm(len(dataset))[: args.n_data]
        ]
    )

    # Load model:
    logger.info("Loading model... ")
    model = model_from_config(config, initialize=True, dataset=dataset)
    model = model.to(device)
    # "Deploy" it
    model = _compile_for_deploy(model)

    logger.info("Starting...")
    t = Timer(stmt="model(next(datas))", globals={"model": model, "datas": datas})

    perloop = t.timeit(args.n)

    print()
    print(perloop)


if __name__ == "__main__":
    main(running_as_script=True)

