import argparse
import textwrap
import tempfile
import itertools
import time
import pdb
import traceback
import pickle

import torch
from torch.utils.benchmark import Timer, Measurement
from torch.utils.benchmark.utils.common import trim_sigfig, select_unit

from e3nn.util.jit import script

from nequip.utils import get_current_code_versions, RankedLogger
from nequip.utils._global_options import _set_global_options, _latest_global_config
from nequip.utils.test import assert_AtomicData_equivariant
from nequip.data import AtomicDataDict
from nequip.data.datamodule import NequIPDataModule
from nequip.scripts.deploy import _compile_for_deploy, load_deployed_model

from omegaconf import OmegaConf
from hydra.utils import instantiate

# TODO: add model-debug-mode


logger = RankedLogger(__name__, rank_zero_only=True)


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
        default=None,
    )
    parser.add_argument(
        "--num_batch",
        help="Number of batches to use (batch size is controlled by the datamodule's train dataloader arguments).",
        type=int,
        default=2,
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

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    config = OmegaConf.load(args.config)
    _ = get_current_code_versions()
    logger.debug("Setting global options ...")
    _set_global_options(**OmegaConf.to_container(config.global_options, resolve=True))

    # === instantiate datamodule ===
    logger.info("Building datamodule ...")

    # == silently include type_names in stats_manager if present ==
    assert "type_names" in config.training_module.model
    data = OmegaConf.to_container(config.data, resolve=True)
    if "stats_manager" in data:
        data["stats_manager"]["type_names"] = config.training_module.model.type_names
    datamodule = instantiate(data, _recursive_=False)
    assert isinstance(datamodule, NequIPDataModule)

    # === compute dataset statistics and use resolver to get dataset statistics to model config ===
    dataset_stats_time = time.time()
    stats_dict = datamodule.get_statistics(dataset="train")
    dataset_stats_time = time.time() - dataset_stats_time
    print(f"Train dataset statistics computation took {dataset_stats_time:.4f}s")

    print("Train dataset statistics:")
    for k, v in stats_dict.items():
        print(f"{k:^30}: {v}")

    def training_data_stats(stat_name: str):
        stat = stats_dict.get(stat_name, None)
        if stat is None:
            raise RuntimeError(
                f"Data statistics field `{stat_name}` was requested for use in model initialization, but was not computed -- users must explicitly configure its computation with the `stats_manager` DataModule argument."
            )
        return stat

    OmegaConf.register_new_resolver(
        "training_data_stats",
        training_data_stats,
        use_cache=True,
    )
    nequip_module_cfg = OmegaConf.to_object(config.training_module)

    # === get smaller data list for testing ===
    datas_list = []
    try:
        datamodule.prepare_data()
        datamodule.setup(stage="fit")
        dloader = datamodule.train_dataloader()
        for data in dloader:
            if len(datas_list) == args.num_batch:
                break
            datas_list.append(AtomicDataDict.to_(data, device))
    finally:
        datamodule.teardown(stage="fit")

    # cycle over the datas we loaded
    datas = itertools.cycle(datas_list)

    # short circut
    if args.n == 0:
        print("Got -n 0, so quitting without running benchmark.")
        return
    elif args.n is None:
        args.n = 5 if args.profile else 30

    # === instantiate NequIP Lightning module ===

    if args.model is None:
        print("Building model and training modules ... ")
        model_time = time.time()
        try:
            nequip_module = instantiate(
                nequip_module_cfg,
                # ensure lazy instantiation of lightning module attributes
                _recursive_=False,
                # make everything Python primitives (no DictConfig/ListConfig)
                _convert_="all",
                num_datasets=datamodule.num_datasets,
            )
            model = nequip_module.model
        except:  # noqa: E722
            if args.pdb:
                traceback.print_exc()
                pdb.post_mortem()
            else:
                raise
        model_time = time.time() - model_time
        print(f"    building model and training modules took {model_time:.4f}s")
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
    warmup = _latest_global_config["_jit_bailout_depth"] + 4  # just to be safe...

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
                out = model(next(datas).copy())
                out[AtomicDataDict.TOTAL_ENERGY_KEY].item()
                p.step()

        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=100))
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
        if args.memory_summary and torch.cuda.is_available():
            torch.cuda.memory._record_memory_history(
                True,
                # keep 100,000 alloc/free events from before the snapshot
                trace_alloc_max_entries=100000,
                # record stack information for the trace events
                trace_alloc_record_context=True,
            )
        print("Warmup...")
        warmup_time = time.time()
        for _ in range(warmup):
            model(next(datas).copy())
        warmup_time = time.time() - warmup_time
        print(f"    {warmup} calls of warmup took {warmup_time:.4f}s")

        print("Benchmarking...")

        # just time
        t = Timer(
            stmt="model(next(datas).copy())",
            globals={"model": model, "datas": datas},
        )
        perloop: Measurement = t.timeit(args.n)

        if args.memory_summary and torch.cuda.is_available():
            print("Memory usage summary:")
            print(torch.cuda.memory_summary())
            snapshot = torch.cuda.memory._snapshot()

            with open("snapshot.pickle", "wb") as f:
                pickle.dump(snapshot, f)

        print(" -- Results --")
        print(
            f"PLEASE NOTE: these are speeds for the MODEL, evaluated on --num_batch={args.num_batch} configurations kept in memory."
        )
        print(
            "A variety of factors affect the performance in real molecular dynamics calculations:"
        )
        print(
            "!!! Molecular dynamics speeds should be measured in LAMMPS; speeds from nequip-benchmark should only be used as an estimate of RELATIVE speed among different hyperparameters."
        )
        print(
            "Please further note that relative speed ordering of hyperparameters is NOT NECESSARILY CONSISTENT across different classes of GPUs (i.e. A100 vs V100 vs consumer) or GPUs vs CPUs."
        )
        print()
        trim_time = trim_sigfig(perloop.times[0], perloop.significant_figures)
        time_unit, time_scale = select_unit(trim_time)
        time_str = ("{:.%dg}" % perloop.significant_figures).format(
            trim_time / time_scale
        )
        print(f"The average call took {time_str}{time_unit}")


if __name__ == "__main__":
    main()
