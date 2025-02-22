import pytest
import torch
from conftest import _check_and_print
import pathlib
import subprocess


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_dimer_plot_example(fake_model_training_session, device):
    """
    Tests that the `plot_dimers` example runs.
    """
    _, tmpdir, env = fake_model_training_session
    path_to_this_file = pathlib.Path(__file__)
    script_path = path_to_this_file.parents[2] / "examples/plot_dimers.py"
    ckpt_path = pathlib.Path(f"{tmpdir}/last.ckpt")
    retcode = subprocess.run(
        [
            "python3",
            script_path,
            "--ckpt-path",
            f"{str(ckpt_path)}",
            "--device",
            device,
            "--output",
            tmpdir + "/dimer_plot_test.png",
        ],
        cwd=tmpdir,
        env=env,
    )
    _check_and_print(retcode)


def test_parity_plot_example(fake_model_training_session):
    """
    Tests that the `parity_plot` example runs.
    """
    _, tmpdir, env = fake_model_training_session
    path_to_this_file = pathlib.Path(__file__)
    script_path = path_to_this_file.parents[2] / "examples/parity_plot.py"
    xyz_path = pathlib.Path(f"{tmpdir}/test_dataset0.xyz")
    retcode = subprocess.run(
        [
            "python3",
            script_path,
            f"{str(xyz_path)}",
            "--output",
            tmpdir + "/parity_plot_test.png",
        ],
        cwd=tmpdir,
        env=env,
    )
    _check_and_print(retcode)
