import pytest
import tempfile
import pathlib
import yaml
import subprocess
import os
import sys

import torch

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


def _check_and_print(retcode):
    __tracebackhide__ = True
    if retcode.returncode:
        if len(retcode.stdout) > 0:
            print(retcode.stdout.decode("ascii"))
        if len(retcode.stderr) > 0:
            print(retcode.stderr.decode("ascii"), file=sys.stderr)
        retcode.check_returncode()


class IdentityModel(GraphModuleMixin, torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._init_irreps(
            irreps_in={
                AtomicDataDict.TOTAL_ENERGY_KEY: "0e",
                AtomicDataDict.FORCE_KEY: "1o",
            },
        )
        self.zero = torch.nn.Parameter(torch.as_tensor(0.0))

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        err = self.zero
        data[AtomicDataDict.FORCE_KEY] = data[AtomicDataDict.FORCE_KEY] + err
        data[AtomicDataDict.NODE_FEATURES_KEY] = (
            0.77 * data[AtomicDataDict.FORCE_KEY].tanh()
        )  # some BS
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = (
            data[AtomicDataDict.TOTAL_ENERGY_KEY] + err
        )
        return data


class ConstFactorModel(GraphModuleMixin, torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._init_irreps(
            irreps_in={
                AtomicDataDict.TOTAL_ENERGY_KEY: "0e",
                AtomicDataDict.FORCE_KEY: "1o",
            },
        )
        # to keep the optimizer happy:
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("factor", 3.7777 * torch.randn(1).squeeze())

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[AtomicDataDict.FORCE_KEY] = (
            self.factor * data[AtomicDataDict.FORCE_KEY] + 0.0 * self.dummy
        )
        data[AtomicDataDict.NODE_FEATURES_KEY] = (
            0.77 * data[AtomicDataDict.FORCE_KEY].tanh()
        )  # some BS
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = (
            self.factor * data[AtomicDataDict.TOTAL_ENERGY_KEY] + 0.0 * self.dummy
        )
        return data


class LearningFactorModel(GraphModuleMixin, torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._init_irreps(
            irreps_in={
                AtomicDataDict.TOTAL_ENERGY_KEY: "0e",
                AtomicDataDict.FORCE_KEY: "1o",
            },
        )
        # By using a big factor, we keep it in a nice descending part
        # of the optimization without too much oscilation in loss at
        # the beginning
        self.factor = torch.nn.Parameter(torch.as_tensor(1.111))

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[AtomicDataDict.FORCE_KEY] = self.factor * data[AtomicDataDict.FORCE_KEY]
        data[AtomicDataDict.NODE_FEATURES_KEY] = (
            0.77 * data[AtomicDataDict.FORCE_KEY].tanh()
        )  # some BS
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = (
            self.factor * data[AtomicDataDict.TOTAL_ENERGY_KEY]
        )
        return data


def _training_session(conffile, model_dtype, builder, BENCHMARK_ROOT):
    default_dtype = str(torch.get_default_dtype())[len("torch.") :]
    if default_dtype == "float32" and model_dtype == "float64":
        pytest.skip("default_dtype=float32 and model_dtype=float64 doesn't make sense")

    path_to_this_file = pathlib.Path(__file__)
    config_path = path_to_this_file.parents[2] / f"configs/{conffile}"
    true_config = yaml.load(config_path.read_text(), Loader=yaml.Loader)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save time
        run_name = "test_train_" + default_dtype
        true_config["run_name"] = run_name
        true_config["root"] = "./"
        true_config["dataset_file_name"] = str(
            BENCHMARK_ROOT / "aspirin_ccsd-train.npz"
        )
        true_config["default_dtype"] = default_dtype
        true_config["model_dtype"] = model_dtype
        true_config["max_epochs"] = 2
        true_config["model_builders"] = [builder]
        # just do forces, which is what the mock models have:
        true_config["loss_coeffs"] = "forces"
        # We need truth labels as inputs for these fake testing models
        true_config["model_input_fields"] = {
            AtomicDataDict.FORCE_KEY: "1o",
            AtomicDataDict.TOTAL_ENERGY_KEY: "0e",
        }

        config_path = tmpdir + "/conf.yaml"
        with open(config_path, "w+") as fp:
            yaml.dump(true_config, fp)
        # == Train model ==
        env = dict(os.environ)
        # make this script available so model builders can be loaded
        env["PYTHONPATH"] = ":".join(
            [str(path_to_this_file.parent)] + env.get("PYTHONPATH", "").split(":")
        )

        retcode = subprocess.run(
            ["nequip-train", "conf.yaml"],
            cwd=tmpdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _check_and_print(retcode)

        yield true_config, tmpdir, env


@pytest.fixture(
    scope="session",
    params=[
        ("minimal.yaml", AtomicDataDict.FORCE_KEY),
        ("minimal_toy_emt.yaml", AtomicDataDict.STRESS_KEY),
    ],
)
def conffile(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=["float32", "float64"],
)
def model_dtype(request, float_tolerance):
    if torch.get_default_dtype() == torch.float32 and model_dtype == "float64":
        pytest.skip("default_dtype=float32 and model_dtype=float64 doesn't make sense")
    return request.param


@pytest.fixture(
    scope="session", params=[ConstFactorModel, LearningFactorModel, IdentityModel]
)
def fake_model_training_session(request, BENCHMARK_ROOT, conffile, model_dtype):
    conffile, _ = conffile
    builder = request.param

    session = _training_session(conffile, model_dtype, builder, BENCHMARK_ROOT)
    true_config, tmpdir, env = next(session)
    yield builder, true_config, tmpdir, env
    del session
