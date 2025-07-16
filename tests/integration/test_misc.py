import pytest
import tempfile
import torch
from omegaconf import OmegaConf
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
    _, tmpdir, env, _ = fake_model_training_session
    path_to_this_file = pathlib.Path(__file__)
    script_path = path_to_this_file.parents[2] / "misc/plot_dimers.py"
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
    _, tmpdir, env, _ = fake_model_training_session
    path_to_this_file = pathlib.Path(__file__)
    script_path = path_to_this_file.parents[2] / "misc/parity_plot.py"
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


def test_lmdb_example():
    """
    Tests that the `lmdb` example runs.
    """
    path_to_this_file = pathlib.Path(__file__)
    example_dir = str(path_to_this_file.parents[2] / "misc/lmdb_dataset_conversion")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = OmegaConf.load(example_dir + "/data.yaml")
        config.file_path = f"{tmpdir}/lmdb_example"
        config = OmegaConf.create(config)
        OmegaConf.save(config=config, f=tmpdir + "/conf.yaml")
        retcode = subprocess.run(
            [
                "python3",
                example_dir + "/datamodule_to_lmdb.py",
                "-cp",
                tmpdir,
                "-cn",
                "conf",
            ],
            cwd=tmpdir,
        )
        _check_and_print(retcode)

        # test ASE example
        # implicitly tests that the tutorial example link isn't broken
        retcode = subprocess.run(
            [
                "python3",
                example_dir + "/example_lmdb_conversion.py",
            ],
            cwd=tmpdir,
        )
        _check_and_print(retcode)
