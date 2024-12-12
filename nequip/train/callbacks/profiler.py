from typing import Optional, Dict, Any
import os
import sys
from pathlib import Path

import torch
import lightning
from lightning.pytorch.callbacks import Callback
from nequip.data import AtomicDataDict
from nequip.train import NequIPLightningModule

_default_schedule_kwargs = dict(wait=5, warmup=1, active=3, repeat=2)


class Profiler(Callback):
    """Proxy class for `Pytorch Profiler <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_.

    Example usage in config:
    ::

        trainer:
          ...
          callbacks:
          - _target_: nequip.train.callbacks.Profiler
            trace_output: "./proflog"

    Args:
        trace_output (str): directory where profile data is stored
        schedule_kwargs (Dict[str,int]): wait/warmup/active/repeat counts for the torch.profiler.schedule
        sort_by (str): argument to profile.key_averages().table() -- e.g. cpu_memory_usage/cpu_time_total/cuda_time_total/xpu_time_total
        key_averages_kwargs (Dict[str,Any]): keyword args to pass to profiler.key_averages()
        row_limit (int): number of "top" functions in profile
        tensorboard_trace (bool): also write outputs using torch.profiler.tensorboard_trace_handler?
    """

    def __init__(
        self,
        trace_output: str = "proflog",
        schedule_kwargs=_default_schedule_kwargs,
        sort_by: str = "cpu_time_total",
        key_averages_kwargs: Dict[str, Any] = {},
        row_limit: int = 10,
        tensorboard_trace: bool = False,
    ) -> None:
        super().__init__()
        self.trace_output = trace_output
        self.sort_by = sort_by
        self.key_averages_kwargs = key_averages_kwargs
        self.row_limit = row_limit

        self.tensorboard_trace = tensorboard_trace
        self.prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(**schedule_kwargs),
            on_trace_ready=self.trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    def show_averages(self, prof):
        avgs = prof.key_averages(**self.key_averages_kwargs)
        try:
            ans = avgs.table(sort_by=self.sort_by, row_limit=self.row_limit)
        except AttributeError as e:
            print(f"Profile key_averages.table - AttributeError: {e}")
            ans = avgs.table(sort_by="cpu_time_total", row_limit=self.row_limit)
        if self.rank == 0:
            print(ans)
        return ans

    def trace_handler(self, prof):
        if self.tensorboard_trace:
            torch.profiler.tensorboard_trace_handler(self.trace_output)

        s = self.show_averages(prof)
        Path(self.trace_output).mkdir(exist_ok=True, parents=True)
        try:
            (
                Path(self.trace_output) / f"profile_{self.rank}_{self.step_num}.txt"
            ).write_text(s)
            # prof.export_chrome_trace(
            #    os.path.join(
            #        self.trace_output, f"trace_{self.rank}_{self.step_num}.json"
            #    )
            # )
            # ERROR:2024-12-12 16:43:09 200183:200183 output_json.cpp:617] Failed to rename proflog/trace_0_17.json.tmp to proflog/trace_0_17.json : No such file or directory
        except Exception as e:
            print("Error writing profile output: " + str(e), file=sys.stderr)

    def on_train_start(self, trainer, pl_module) -> None:
        self.rank = pl_module.global_rank
        self.prof.start()

    def on_train_end(self, trainer, pl_module) -> None:
        self.prof.stop()

    def on_train_batch_start(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
        batch: AtomicDataDict.Type,
        batch_idx: int,
    ) -> None:
        """"""
        self.step_num = pl_module.global_step
        self.prof.step()
