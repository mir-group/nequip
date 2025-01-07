import pytest
import tempfile
import pathlib
import subprocess
import os
import sys
from omegaconf import OmegaConf, open_dict


def _check_and_print(retcode):
    __tracebackhide__ = True
    if retcode.returncode:
        if retcode.stdout is not None and len(retcode.stdout) > 0:
            print(retcode.stdout.decode("ascii"))
        if retcode.stderr is not None and len(retcode.stderr) > 0:
            print(retcode.stderr.decode("ascii"), file=sys.stderr)
        retcode.check_returncode()


def _training_session(conffile, model_dtype, BENCHMARK_ROOT):
    path_to_this_file = pathlib.Path(__file__)
    config_path = path_to_this_file.parents[2] / f"configs/{conffile}"
    config = OmegaConf.load(config_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        with tempfile.TemporaryDirectory() as data_tmpdir:
            if "data_source_dir" in config.data:
                config.data.data_source_dir = data_tmpdir
            config.training_module.model.model_dtype = model_dtype
            config.data.val_dataloader_kwargs.batch_size = 1
            with open_dict(config):
                config["hydra"] = {"run": {"dir": tmpdir}}
            config = OmegaConf.create(config)
            config_path = tmpdir + "/conf.yaml"
            OmegaConf.save(config=config, f=config_path)

            # == Train model ==
            env = dict(os.environ)
            retcode = subprocess.run(
                ["nequip-train", "-cn", "conf"],
                cwd=tmpdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _check_and_print(retcode)
            yield config, tmpdir, env


@pytest.fixture(
    scope="session",
    params=["minimal.yaml", "minimal_toy_emt.yaml"],
)
def conffile(request):
    return request.param


@pytest.fixture(scope="session")
def fake_model_training_session(BENCHMARK_ROOT, conffile, model_dtype):
    session = _training_session(conffile, model_dtype, BENCHMARK_ROOT)
    config, tmpdir, env = next(session)
    yield config, tmpdir, env
    del session
