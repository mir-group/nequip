import os

_has_sched_getaffinity: bool = hasattr(os, "sched_getaffinity")


def num_tasks() -> int:
    # sched_getaffinity gives number of _allowed_ cores
    # this is correct for SLURM jobs, for example
    num_avail: int
    if _has_sched_getaffinity:
        num_avail = len(os.sched_getaffinity(0))
    else:
        # on macOS, at least, sched_getaffinity() doesn't appear to be available.
        # fallback to something sane
        num_avail = os.cpu_count()
    # If we couldn't get affinity, don't default to the whole system... sane default to 1
    n_proc: int = int(
        os.environ.get("NEQUIP_NUM_TASKS", num_avail if _has_sched_getaffinity else 1)
    )
    assert n_proc > 0
    assert (
        n_proc <= num_avail
    ), f"Asked for more worker tasks NEQUIP_NUM_TASKS={n_proc} than available CPU cores {num_avail}"
    return n_proc
