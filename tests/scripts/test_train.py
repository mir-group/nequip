import pytest
import tempfile
import pathlib
import yaml
import subprocess
import os

import torch

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

# from nequip.scripts import train


class IdentityModel(GraphModuleMixin, torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._init_irreps(
            irreps_in={
                AtomicDataDict.TOTAL_ENERGY_KEY: "0e",
                AtomicDataDict.FORCE_KEY: "1o",
            }
        )
        self.one = torch.nn.Parameter(torch.as_tensor(1.0))

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[AtomicDataDict.FORCE_KEY] = self.one * data[AtomicDataDict.FORCE_KEY]
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = (
            self.one * data[AtomicDataDict.TOTAL_ENERGY_KEY]
        )
        return data


@pytest.mark.parametrize(
    "conffile,field",
    [
        ("minimal.yaml", AtomicDataDict.FORCE_KEY),
        ("minimal_eng.yaml", AtomicDataDict.TOTAL_ENERGY_KEY),
    ],
)
def test_identity_train(nequip_dataset, BENCHMARK_ROOT, conffile, field):

    dtype = str(torch.get_default_dtype())[len("torch.") :]

    if torch.cuda.is_available():
        # TODO: is this true?
        pytest.skip("CUDA and subprocesses have issues")

    path_to_this_file = pathlib.Path(__file__)
    config_path = path_to_this_file.parents[2] / f"configs/{conffile}"
    true_config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save time
        run_name = "test_train" + dtype
        true_config["run_name"] = run_name
        true_config["root"] = tmpdir
        true_config["dataset_file_name"] = str(
            BENCHMARK_ROOT / "aspirin_ccsd-train.npz"
        )
        true_config["default_dtype"] = dtype
        true_config["max_epochs"] = 1
        true_config["model_builder"] = IdentityModel
        config_path = tmpdir + "/conf.yaml"
        with open(config_path, "w+") as fp:
            yaml.dump(true_config, fp)
        # Train model
        env = dict(os.environ)
        # make this script available so model builders can be loaded
        env["PYTHONPATH"] = ":".join(
            [str(path_to_this_file.parent)] + env.get("PYTHONPATH", "").split(":")
        )
        retcode = subprocess.run(
            ["nequip-train", str(config_path)], cwd=tmpdir, env=env
        )
        retcode.check_returncode()
        # TODO: check metrics from training run!
