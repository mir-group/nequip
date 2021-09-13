import sys
import argparse
import textwrap
import tempfile
import contextlib
import itertools

import torch
from torch.utils.benchmark import Timer, Measurement
from torch.utils.benchmark.utils.common import trim_sigfig, select_unit

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicData
from nequip.model import model_from_config
from nequip.scripts.deploy import _compile_for_deploy
from nequip.scripts.train import _set_global_options, default_config


def main(args=None):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Benchmark the approximate MD performance of a given model configuration / dataset pair."""
        )
    )
    parser.add_argument("config", help="configuration file")
    parser.add_argument(
        "--profile",
        help="Profile instead of timing, creating and outputing a Chrome trace JSON to the given path.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--device",
        help="Device to run the model on. If not provided, defaults to CUDA if available and CPU otherwise.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-n",
        help="Number of trials.",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--n-data",
        help="Number of frames to use.",
        type=int,
        default=1,
    )

    # TODO: option to profile
    # TODO: option to show memory use

    # Parse the args
    args = parser.parse_args(args=args)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    config = Config.from_file(args.config, defaults=default_config)
    _set_global_options(config)

    # Load dataset to get something to benchmark on
    print("Loading dataset... ")
    # Currently, pytorch_geometric prints some status messages to stdout while loading the dataset
    # TODO: fix may come soon: https://github.com/rusty1s/pytorch_geometric/pull/2950
    # Until it does, just redirect them.
    with contextlib.redirect_stdout(sys.stderr):
        dataset = dataset_from_config(config)
    datas = [
        AtomicData.to_AtomicDataDict(dataset[i].to(device))
        for i in torch.randperm(len(dataset))[: args.n_data]
    ]
    n_atom: int = len(datas[0]["pos"])
    assert all(len(d["pos"]) == n_atom for d in datas)  # TODO handle the general case
    # TODO: show some stats about datas

    datas = itertools.cycle(datas)

    # Load model:
    print("Loading model... ")
    model = model_from_config(config, initialize=True, dataset=dataset)
    model = model.to(device)
    print("Compile...")
    # "Deploy" it
    model = _compile_for_deploy(model)
    # save and reload to avoid bugs
    with tempfile.NamedTemporaryFile() as f:
        torch.jit.save(model, f.name)
        model = torch.jit.load(f.name)

    # Make sure we're warm past compilation
    warmup = config["_jit_bailout_depth"]

    print("Starting...")
    if args.profile is not None:

        def trace_handler(p):
            p.export_chrome_trace(args.profile)
            print(f"Wrote profiling trace to `{args.profile}`")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
            ]
            + (
                [torch.profiler.ProfilerActivity.CUDA]
                if torch.cuda.is_available()
                else []
            ),
            schedule=torch.profiler.schedule(
                wait=1, warmup=warmup, active=args.n, repeat=1
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for _ in range(1 + warmup + args.n):
                model(next(datas))
                p.step()
    else:
        print("Warmup...")
        for _ in range(warmup):
            model(next(datas))

        # just time
        t = Timer(stmt="model(next(datas))", globals={"model": model, "datas": datas})
        perloop: Measurement = t.timeit(args.n)

        print(" -- Results --")
        print(
            f"PLEASE NOTE: these are speeds for the MODEL, evaluated on --n-data={args.n_data} configurations kept in memory."
        )
        print(
            "    \\_ MD itself, memory copies, and other overhead will affect real-world performance."
        )
        print()
        trim_time = trim_sigfig(perloop.times[0], perloop.significant_figures)
        time_unit, time_scale = select_unit(trim_time)
        time_str = ("{:.%dg}" % perloop.significant_figures).format(
            trim_time / time_scale
        )
        print(f"The average call took {time_str}{time_unit}")
        print(
            "Assuming linear scaling — which is ALMOST NEVER true in practice, especially on GPU —"
        )
        print(
            f"    \\_ this comes out to {trim_time / n_atom:.2f} {time_unit}/atom/call"
        )
        ns_day = (86400.0 / trim_time) * 2e-6
        #     day in s^   step in s^       ^ 2fs / ns
        print(
            f"For this system, at a 2fs timestep, this comes out to {ns_day:.2f} ns/day"
        )


if __name__ == "__main__":
    main()
