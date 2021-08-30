import pytest
import tempfile
import pathlib
import yaml
import subprocess
import os

import numpy as np
import torch

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, RescaleOutput


class IdentityModel(GraphModuleMixin, torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._init_irreps(
            irreps_in={
                AtomicDataDict.TOTAL_ENERGY_KEY: "0e",
                AtomicDataDict.FORCE_KEY: "1o",
            },
        )
        self.one = torch.nn.Parameter(torch.as_tensor(1.0))

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[AtomicDataDict.FORCE_KEY] = self.one * data[AtomicDataDict.FORCE_KEY]
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = (
            self.one * data[AtomicDataDict.TOTAL_ENERGY_KEY]
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
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = (
            self.factor * data[AtomicDataDict.TOTAL_ENERGY_KEY]
        )
        return data


@pytest.mark.parametrize(
    "conffile,field",
    [
        ("minimal.yaml", AtomicDataDict.FORCE_KEY),
        ("minimal_eng.yaml", AtomicDataDict.TOTAL_ENERGY_KEY),
    ],
)
@pytest.mark.parametrize(
    "builder", [IdentityModel, ConstFactorModel, LearningFactorModel]
)
def test_metrics(nequip_dataset, BENCHMARK_ROOT, conffile, field, builder):

    dtype = str(torch.get_default_dtype())[len("torch.") :]

    # if torch.cuda.is_available():
    #     # TODO: is this true?
    #     pytest.skip("CUDA and subprocesses have issues")

    path_to_this_file = pathlib.Path(__file__)
    config_path = path_to_this_file.parents[2] / f"configs/{conffile}"
    true_config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save time
        run_name = "test_train_" + dtype
        true_config["run_name"] = run_name
        true_config["root"] = tmpdir
        true_config["dataset_file_name"] = str(
            BENCHMARK_ROOT / "aspirin_ccsd-train.npz"
        )
        true_config["default_dtype"] = dtype
        true_config["max_epochs"] = 2
        # We just don't add rescaling:
        true_config["model_builders"] = [builder]

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
            ["nequip-train", str(config_path)], cwd=tmpdir, env=env
        )
        retcode.check_returncode()

        # == Load metrics ==
        outdir = f"{true_config['root']}/{true_config['run_name']}/"

        if builder == IdentityModel or builder == LearningFactorModel:
            for which in ("train", "val"):
                dat = np.genfromtxt(
                    f"{outdir}/metrics_batch_{which}.csv",
                    delimiter=",",
                    names=True,
                    dtype=None,
                )
                for field in dat.dtype.names:
                    if field == "epoch" or field == "batch":
                        continue
                    # Everything else should be a loss or a metric
                    if builder == IdentityModel:
                        assert np.allclose(
                            dat[field], 0.0
                        ), f"Loss/metric `{field}` wasn't all zeros for {which}"
                    elif builder == LearningFactorModel:
                        assert (
                            dat[field][-1] < dat[field][0]
                        ), f"Loss/metric `{field}` didn't go down for {which}"

        # epoch metrics
        dat = np.genfromtxt(
            f"{outdir}/metrics_epoch.csv",
            delimiter=",",
            names=True,
            dtype=None,
        )
        for field in dat.dtype.names:
            if field == "epoch" or field == "wall" or field == "LR":
                continue
            # Everything else should be a loss or a metric
            if builder == IdentityModel:
                assert np.allclose(
                    dat[field], 0.0
                ), f"Loss/metric `{field}` wasn't all equal to zero for epoch"
            elif builder == ConstFactorModel:
                # otherwise just check its constant.
                # epoch-wise numbers should be the same, since there's no randomness at this level
                assert np.allclose(
                    dat[field], dat[field][0]
                ), f"Loss/metric `{field}` wasn't all equal to {dat[field][0]} for epoch"
            elif builder == LearningFactorModel:
                assert (
                    dat[field][-1] < dat[field][0]
                ), f"Loss/metric `{field}` didn't go down across epochs"

        # == Check model ==
        model = torch.load(outdir + "/last_model.pth")
        assert isinstance(
            model, builder
        )  # make sure trainer and this test aren't out of sync

        if builder == IdentityModel:
            one = model.one
            # Since the loss is always zero, even though the constant
            # 1 was trainable, it shouldn't have changed
            assert torch.allclose(
                one, torch.ones(1, device=one.device, dtype=one.dtype)
            )
