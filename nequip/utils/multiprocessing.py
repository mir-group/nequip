import os


def num_tasks() -> int:
    # sched_getaffinity gives number of _allowed_ cores
    # this is correct for SLURM jobs, for example
    num_avail: int = len(os.sched_getaffinity(0))
    n_proc: int = int(os.environ.get("NEQUIP_NUM_TASKS", num_avail))
    assert n_proc > 0
    assert (
        n_proc <= num_avail
    ), f"Asked for more worker tasks NEQUIP_NUM_TASKS={n_proc} than available CPU cores {num_avail}"
    return n_proc
