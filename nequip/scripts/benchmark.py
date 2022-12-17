import argparse
import textwrap
import tempfile
import itertools
import time
import logging
import sys
import pdb
import traceback

import torch
from torch.utils.benchmark import Timer, Measurement
from torch.utils.benchmark.utils.common import trim_sigfig, select_unit

from e3nn.util.jit import script

from nequip.utils import Config
from nequip.utils.test import assert_AtomicData_equivariant
from nequip.data import AtomicData, AtomicDataDict, dataset_from_config
from nequip.model import model_from_config
from nequip.scripts.deploy import _compile_for_deploy, load_deployed_model
from nequip.scripts.train import default_config, check_code_version
from nequip.utils._global_options import _set_global_options


def main(args=None):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Benchmark the approximate MD performance of a given model configuration / dataset pair."""
        )
    )
    parser.add_argument("config", help="configuration file")
    parser.add_argument(
        "--model",
        help="A deployed model to load instead of building a new one from `config`. ",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--profile",
        help="Profile instead of timing, creating and outputing a Chrome trace JSON to the given path.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--equivariance-test",
        help="test the model's equivariance on `--n-data` frames.",
        action="store_true",
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
    parser.add_argument(
        "--timestep",
        help="MD timestep for ns/day esimation, in fs. Defauts to 1fs.",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--no-compile",
        help="Don't compile the model to TorchScript",
        action="store_true",
    )
    parser.add_argument(
        "--memory-summary",
        help="Print torch.cuda.memory_summary() after running the model",
        action="store_true",
    )
    parser.add_argument(
        "--verbose", help="Logging verbosity level", type=str, default="error"
    )
    parser.add_argument(
        "--pdb",
        help="Run model builders and model under debugger to easily drop to debugger to investigate errors.",
        action="store_true",
    )

    # Parse the args
    args = parser.parse_args(args=args)
    if args.pdb:
        assert args.profile is None

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, args.verbose.upper()))
    root_logger.handlers = [logging.StreamHandler(sys.stderr)]

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    config = Config.from_file(args.config, defaults=default_config)
    _set_global_options(config)
    check_code_version(config)

    # Load dataset to get something to benchmark on
    print("Loading dataset... ")
    dataset_time = time.time()
    dataset = dataset_from_config(config)
    dataset_time = time.time() - dataset_time
    print(f"    loading dataset took {dataset_time:.4f}s")
    dataset_rng = torch.Generator()
    dataset_rng.manual_seed(config.get("dataset_seed", config.get("seed", 12345)))
    datas_list = [
        AtomicData.to_AtomicDataDict(dataset[i].to(device))
        for i in torch.randperm(len(dataset), generator=dataset_rng)[: args.n_data]
    ]
    n_atom: int = len(datas_list[0]["pos"])
    if not all(len(d["pos"]) == n_atom for d in datas_list):
        raise NotImplementedError(
            "nequip-benchmark does not currently handle benchmarking on data frames with variable number of atoms"
        )
    print(
        f"    loaded dataset of size {len(dataset)} and sampled --n-data={args.n_data} frames"
    )
    # print some dataset information
    print("    benchmark frames statistics:")
    print(f"         number of atoms: {n_atom}")
    print(f"         number of types: {dataset.type_mapper.num_types}")
    print(
        f"          avg. num edges: {sum(d[AtomicDataDict.EDGE_INDEX_KEY].shape[1] for d in datas_list) / len(datas_list)}"
    )
    avg_edges_per_atom = torch.mean(
        torch.cat(
            [
                torch.bincount(
                    d[AtomicDataDict.EDGE_INDEX_KEY][0],
                    minlength=d[AtomicDataDict.POSITIONS_KEY].shape[0],
                ).float()
                for d in datas_list
            ]
        )
    ).item()
    print(f"         avg. neigh/atom: {avg_edges_per_atom}")

    # cycle over the datas we loaded
    datas = itertools.cycle(datas_list)

    # short circut
    if args.n == 0:
        print("Got -n 0, so quitting without running benchmark.")
        return

    # Load model:
    if args.model is None:
        print("Building model... ")
        model_time = time.time()
        try:
            model = model_from_config(
                config, initialize=True, dataset=dataset, deploy=True
            )
        except:  # noqa: E722
            if args.pdb:
                traceback.print_exc()
                pdb.post_mortem()
            else:
                raise
        model_time = time.time() - model_time
        print(f"    building model took {model_time:.4f}s")
    else:
        print("Loading model...")
        model, metadata = load_deployed_model(args.model, device=device, freeze=False)
        print("    deployed model has metadata:")
        print(
            "\n".join(
                "        %s: %s" % e for e in metadata.items() if e[0] != "config"
            )
        )
    print(f"    model has {sum(p.numel() for p in model.parameters())} weights")
    print(
        f"    model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable weights"
    )
    print(
        f"    model weights and buffers take {sum(p.numel() * p.element_size() for p in itertools.chain(model.parameters(), model.buffers())) / (1024 * 1024):.2f} MB"
    )

    model.eval()
    if args.equivariance_test:
        args.no_compile = True
        if args.model is not None:
            raise RuntimeError("Can't equivariance test a deployed model.")

    if args.no_compile:
        model = model.to(device)
    else:
        print("Compile...")
        # "Deploy" it
        compile_time = time.time()
        model = script(model)
        model = _compile_for_deploy(model)
        compile_time = time.time() - compile_time
        print(f"    compilation took {compile_time:.4f}s")

        # save and reload to avoid bugs
        with tempfile.NamedTemporaryFile() as f:
            torch.jit.save(model, f.name)
            model = torch.jit.load(f.name, map_location=device)
            # freeze like in the LAMMPS plugin
            model = torch.jit.freeze(model)
            # and reload again just to avoid bugs
            torch.jit.save(model, f.name)
            model = torch.jit.load(f.name, map_location=device)

    # Make sure we're warm past compilation
    warmup = config["_jit_bailout_depth"] + 4  # just to be safe...

    if args.profile is not None:

        def trace_handler(p):
            p.export_chrome_trace(args.profile)
            print(f"Wrote profiling trace to `{args.profile}`")

        print("Starting profiling...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
            ]
            + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
            schedule=torch.profiler.schedule(
                wait=1, warmup=warmup, active=args.n, repeat=1
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for _ in range(1 + warmup + args.n):
                model(next(datas).copy())
                p.step()
    elif args.pdb:
        print("Running model under debugger...")
        try:
            for _ in range(args.n):
                model(next(datas).copy())
        except:  # noqa: E722
            traceback.print_exc()
            pdb.post_mortem()
        print("Done.")
    elif args.equivariance_test:
        print("Warmup...")
        warmup_time = time.time()
        for _ in range(warmup):
            model(next(datas).copy())
        warmup_time = time.time() - warmup_time
        print(f"    {warmup} calls of warmup took {warmup_time:.4f}s")
        print("Running equivariance test...")
        errstr = assert_AtomicData_equivariant(model, datas_list)
        print(
            "    Equivariance test passed; equivariance errors:\n"
            "    Errors are in real units, where relevant.\n"
            "    Please note that the large scale of the typical\n"
            "    shifts to the (atomic) energy can cause\n"
            "    catastrophic cancellation and give incorrectly\n"
            "    the equivariance error as zero for those fields.\n"
            f"{errstr}"
        )
        del errstr
    else:
        print("Warmup...")
        warmup_time = time.time()
        for _ in range(warmup):
            model(next(datas).copy())
        warmup_time = time.time() - warmup_time
        print(f"    {warmup} calls of warmup took {warmup_time:.4f}s")

        print("Benchmarking...")
        # just time
        t = Timer(
            stmt="model(next(datas).copy())", globals={"model": model, "datas": datas}
        )
        perloop: Measurement = t.timeit(args.n)

        if args.memory_summary and torch.cuda.is_available():
            print("Memory usage summary:")
            print(torch.cuda.memory_summary())

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
        per_atom_time = trim_time / n_atom
        time_unit_per, time_scale_per = select_unit(per_atom_time)
        print(
            f"    \\_ this comes out to {per_atom_time/time_scale_per:g} {time_unit_per}/atom/call"
        )
        ns_day = (86400.0 / trim_time) * args.timestep * 1e-6
        #     day in s^   s/step^         ^ fs / step      ^ ns / fs
        print(
            f"For this system, at a {args.timestep:.2f}fs timestep, this comes out to {ns_day:.2f} ns/day"
        )


if __name__ == "__main__":
    main()
