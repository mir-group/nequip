# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import os
import shutil
import socket

import torch
import lightning
from lightning.pytorch.callbacks import Callback

from nequip.train import NequIPLightningModule
from nequip.utils import RankedLogger

from typing import List, Optional, Sequence

logger = RankedLogger(__name__, rank_zero_only=False)

# the TensorBoard profiler plugin (and HTA) discover traces by this suffix,
# and the same file can be loaded directly by `chrome://tracing` or https://ui.perfetto.dev
_TRACE_SUFFIX = ".pt.trace.json"


class TorchProfilerCallback(Callback):
    """Profile training steps with ``torch.profiler`` and export Chrome/TensorBoard traces.

    The callback wraps the training loop in a :class:`torch.profiler.profile` context driven by
    a :func:`torch.profiler.schedule`. Profiling starts *after* Lightning's sanity-check
    validation, and one trace file is written per completed ``wait``/``warmup``/``active``
    cycle. The trace files are named ``<worker>.cycle<N>.pt.trace.json`` and can either be
    loaded directly in ``chrome://tracing`` / `Perfetto <https://ui.perfetto.dev>`_, or pointed
    at by the TensorBoard profiler plugin (``tensorboard --logdir <profile_dir>``).
    To use chrome tracing, make sure to have Chrome's developer settings enabled.

    The profiler samples ``repeat * (wait + warmup + active)`` training batches in total, so
    make sure the run is long enough to reach the end of the first cycle -- otherwise no trace
    is written. A warning is emitted at the start of training if the configured schedule cannot
    complete.

    Example usage in config:

    .. code-block:: yaml

        callbacks:
          - _target_: nequip.train.callbacks.TorchProfilerCallback
            profile_dir: ${hydra:runtime.output_dir}/profiler
            wait: 5
            warmup: 3
            active: 5
            repeat: 1

    .. warning::
        Some of these profiling options are a bit expensive. In particular, set ``active`` small,
        and use ``with_stack`` and ``profile_memory`` only if you need to.

    .. note::
        When using ``compile_mode: compile``, the first several batches are
        dominated by compilation. Set ``wait`` (and/or ``warmup``) large enough to skip past
        compilation, otherwise the trace mostly shows Dynamo/Inductor instead of the model.

    Args:
        profile_dir (str): directory that trace files are written to
        chrome_trace_dir (str): optional additional directory to copy the traces into (default ``None``)
        wait (int): number of batches to skip at the start of each cycle
        warmup (int): number of batches to warm the profiler up on
        active (int): number of batches recorded in each cycle
        repeat (int): number of cycles, ``0`` means repeat forever
        ranks (List[int]): global ranks that profile, ``None`` profiles every rank (default ``[0]``)
        record_shapes (bool): whether to record input shapes of operators
        with_stack (bool): whether to record source information (file and line number)
        with_modules (bool): whether to record the module hierarchy of each operator
        profile_memory (bool): whether to record tensor memory allocation/free
        activities (List[str]): profiler activities, e.g. ``["CPU", "CUDA"]``
            (default ``None``: CPU, if available, CUDA)
        summary_rows (int): number of rows of the ``key_averages`` summary table written next to
            each trace, ``0`` disables the summary (default ``30``)
    """

    def __init__(
        self,
        profile_dir: str = "profiler",
        chrome_trace_dir: Optional[str] = None,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        ranks: Optional[Sequence[int]] = (0,),
        record_shapes: bool = True,
        with_stack: bool = False,
        with_modules: bool = False,
        profile_memory: bool = False,
        activities: Optional[Sequence[str]] = None,
        summary_rows: int = 30,
    ):
        super().__init__()
        assert wait >= 0 and warmup >= 0 and repeat >= 0, (
            "`wait`, `warmup` and `repeat` must be nonnegative"
        )
        assert active >= 1, "`active` must be at least 1"
        if warmup == 0:
            logger.warning(
                "`TorchProfilerCallback` was configured with `warmup=0`, which can skew the "
                "first profiled batches -- `warmup >= 1` is recommended."
            )
        self.profile_dir = profile_dir
        self.chrome_trace_dir = chrome_trace_dir
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.ranks = None if ranks is None else list(ranks)
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.with_modules = with_modules
        self.profile_memory = profile_memory
        # resolve eagerly so that a typo fails at config parsing time, not mid-training
        self.activities = self._resolve_activities(activities)
        self.summary_rows = summary_rows

        self._prof = None
        self._cycle = 0

    # === setup and teardown ===

    def on_train_start(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ) -> None:
        # NOTE: `on_train_start` (as opposed to `on_fit_start`) is used so that the sanity-check
        # validation, which is not part of the steady-state training loop, is not profiled
        if self._prof is not None:
            return
        if self.ranks is not None and trainer.global_rank not in self.ranks:
            logger.debug(
                f"`TorchProfilerCallback` inactive on rank {trainer.global_rank} (profiling ranks {self.ranks})"
            )
            return

        self._warn_if_schedule_unreachable(trainer)

        os.makedirs(self.profile_dir, exist_ok=True)
        if self.chrome_trace_dir is not None:
            os.makedirs(self.chrome_trace_dir, exist_ok=True)

        self._prof = torch.profiler.profile(
            activities=self.activities,
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=self._on_trace_ready,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
            with_modules=self.with_modules,
            profile_memory=self.profile_memory,
        )
        self._prof.__enter__()
        logger.info(
            f"`TorchProfilerCallback` started on rank {trainer.global_rank} "
            f"(wait={self.wait}, warmup={self.warmup}, active={self.active}, repeat={self.repeat}); "
            f"traces will be written to {os.path.abspath(self.profile_dir)}"
        )

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if self._prof is not None:
            self._prof.step()

    def on_train_end(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
    ) -> None:
        self._stop()

    def on_exception(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
        exception: BaseException,
    ) -> None:
        # make sure the profiler is torn down (and any completed cycle flushed) if training dies
        self._stop()

    def teardown(
        self,
        trainer: lightning.Trainer,
        pl_module: NequIPLightningModule,
        stage: str,
    ) -> None:
        # last-resort safety net -- `torch.profiler` is process-global state, so leaving it
        # running would slow down (and pollute) any subsequent `val`/`test` run
        self._stop()

    def _stop(self) -> None:
        if self._prof is None:
            return
        prof, self._prof = self._prof, None
        prof.__exit__(None, None, None)
        if self._cycle == 0:
            logger.warning(
                "`TorchProfilerCallback` did not write any trace -- training ended before the "
                f"first profiling cycle of {self.wait + self.warmup + self.active} batches "
                "completed. Reduce `wait`/`warmup`/`active` or train for more batches."
            )

    # === trace export ===

    def _on_trace_ready(self, prof) -> None:
        cycle = self._cycle
        self._cycle += 1
        # `time_ns` is deliberately not used (unlike `torch.profiler.tensorboard_trace_handler`):
        # hostname + pid already disambiguates ranks, and stable names are easier to script over
        basename = f"{socket.gethostname()}_{os.getpid()}.cycle{cycle}"
        path = os.path.abspath(os.path.join(self.profile_dir, basename + _TRACE_SUFFIX))

        # NOTE: kineto can only serialize a given trace once -- exporting a second time raises
        # "Trace is already saved." -- so we export once here and copy if another copy is wanted
        prof.export_chrome_trace(path)
        logger.info(
            f"`TorchProfilerCallback` wrote trace for cycle {cycle} to {path} "
            f"({os.path.getsize(path) / 1e6:.1f} MB)"
        )

        if self.chrome_trace_dir is not None:
            dest = os.path.abspath(
                os.path.join(self.chrome_trace_dir, basename + _TRACE_SUFFIX)
            )
            if dest != path:
                shutil.copyfile(path, dest)
                logger.info(f"`TorchProfilerCallback` copied trace to {dest}")

        if self.summary_rows > 0:
            self._write_summary(prof, os.path.join(self.profile_dir, basename + ".txt"))

    def _write_summary(self, prof, path: str) -> None:
        sort_by = (
            "self_device_time_total"
            if torch.profiler.ProfilerActivity.CUDA in self.activities
            else "self_cpu_time_total"
        )
        try:
            table = prof.key_averages(group_by_input_shape=False).table(
                sort_by=sort_by, row_limit=self.summary_rows
            )
        except Exception as e:
            # the summary is a convenience -- never let it take down a training run
            logger.warning(
                f"`TorchProfilerCallback` could not build summary table: {e}"
            )
            return
        with open(path, "w") as f:
            f.write(table + "\n")

    # === helpers ===

    @staticmethod
    def _resolve_activities(activities: Optional[Sequence[str]]) -> List:
        if activities is None:
            acts = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                acts.append(torch.profiler.ProfilerActivity.CUDA)
            return acts

        resolved = []
        for a in activities:
            act = getattr(torch.profiler.ProfilerActivity, str(a).upper(), None)
            if act is None:
                # `PrivateUse1` is not upper case, so check the raw name too
                act = getattr(torch.profiler.ProfilerActivity, str(a), None)
            if act is None:
                raise ValueError(
                    f"Unknown profiler activity `{a}` -- options are "
                    f"{[m for m in dir(torch.profiler.ProfilerActivity) if not m.startswith('_')]}"
                )
            resolved.append(act)
        return resolved

    def _warn_if_schedule_unreachable(self, trainer: lightning.Trainer) -> None:
        """Warn early if the run is too short for even one profiling cycle to complete."""
        num_batches = getattr(trainer, "num_training_batches", float("inf"))
        max_epochs = trainer.max_epochs if trainer.max_epochs is not None else -1
        if not (0 < num_batches < float("inf")) or max_epochs <= 0:
            # can't tell (IterableDataset, `max_steps`-driven run, etc.)
            return
        available = num_batches * max_epochs
        if trainer.max_steps is not None and trainer.max_steps > 0:
            available = min(available, trainer.max_steps)
        needed = self.wait + self.warmup + self.active
        if available < needed:
            logger.warning(
                f"`TorchProfilerCallback` needs {needed} training batches to complete one "
                f"profiling cycle, but this run has at most {int(available)} -- no trace will be "
                "written. Reduce `wait`/`warmup`/`active`, or train for more batches."
            )
