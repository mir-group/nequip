"""
Trainer tests
"""
import pytest

import numpy as np
import tempfile
from os.path import isfile

import torch
from torch.nn import Linear

from nequip.model import model_from_config
from nequip.data import AtomicDataDict
from nequip.train.trainer import Trainer
from nequip.utils.savenload import load_file
from nequip.nn import GraphModuleMixin
from nequip.utils import Config


def dummy_builder():
    return DummyNet(3)


# set up two config to test
NATOMS = 3
NFRAMES = 10
minimal_config = Config(
    config=dict(
        run_name="test",
        n_train=4,
        n_val=4,
        exclude_keys=["sth"],
        max_epochs=2,
        batch_size=2,
        learning_rate=1e-2,
        optimizer="Adam",
        seed=0,
        append=False,
        T_0=50,
        T_mult=2,
        loss_coeffs={"forces": 2},
        early_stopping_patiences={"loss": 50},
        early_stopping_lower_bounds={"LR": 1e-10},
        model_builders=[dummy_builder],
        verbose="debug",
        default_dtype=
    )
)


@pytest.fixture(scope="class")
def trainer():
    """
    Generate a class instance with minimal configurations
    """
    minimal_config["default_dtype"] = str(torch.get_default_dtype())[len("torch.") :]
    model = model_from_config(minimal_config)
    with tempfile.TemporaryDirectory(prefix="output") as path:
        minimal_config["root"] = path
        c = Trainer.from_config(model=model, config=minimal_config)
        yield c


class TestTrainerSetUp:
    """
    test initialization
    """

    def test_init(self, trainer):
        assert isinstance(trainer, Trainer)


class TestDuplicateError:
    def test_duplicate_id_2(self, temp_data):
        """
        check whether the Output class can automatically
        insert timestr when a workdir has pre-existed
        """

        minimal_config["root"] = temp_data

        model = DummyNet(3)
        Trainer.from_config(model=model, config=minimal_config)

        with pytest.raises(RuntimeError):
            Trainer.from_config(model=model, config=minimal_config)


class TestSaveLoad:
    @pytest.mark.parametrize("state_dict", [True, False])
    @pytest.mark.parametrize("training_progress", [True, False])
    def test_as_dict(self, trainer, state_dict, training_progress):

        dictionary = trainer.as_dict(
            state_dict=state_dict, training_progress=training_progress
        )

        assert "optimizer_kwargs" in dictionary
        assert state_dict == ("state_dict" in dictionary)
        assert training_progress == ("progress" in dictionary)
        assert len(dictionary["optimizer_kwargs"]) > 1

    @pytest.mark.parametrize("format, suffix", [("torch", "pth"), ("yaml", "yaml")])
    def test_save(self, trainer, format, suffix):

        with tempfile.NamedTemporaryFile(suffix="." + suffix) as tmp:
            file_name = tmp.name
            trainer.save(file_name, format=format)
            assert isfile(file_name), "fail to save to file"
            assert suffix in file_name

    @pytest.mark.parametrize("append", [True])  # , False])
    def test_from_dict(self, trainer, append):

        # torch.save(trainer.model, trainer.best_model_path)

        dictionary = trainer.as_dict(state_dict=True, training_progress=True)
        trainer1 = Trainer.from_dict(dictionary, append=append)

        for key in [
            "best_model_path",
            "last_model_path",
            "logfile",
            "epoch_log",
            "batch_log",
            "workdir",
        ]:
            v1 = getattr(trainer, key, None)
            v2 = getattr(trainer1, key, None)
            assert append == (v1 == v2)

    @pytest.mark.parametrize("append", [True])  # , False])
    def test_from_file(self, trainer, append):

        format = "torch"

        with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:

            file_name = tmp.name
            file_name = trainer.save(file_name, format=format)

            assert isfile(trainer.last_model_path)

            trainer1 = Trainer.from_file(file_name, format=format, append=append)

            for key in [
                "best_model_path",
                "last_model_path",
                "logfile",
                "epoch_log",
                "batch_log",
                "workdir",
            ]:
                v1 = getattr(trainer, key, None)
                v2 = getattr(trainer1, key, None)
                assert append == (v1 == v2)

            for iparam, group1 in enumerate(trainer.optim.param_groups):
                group2 = trainer1.optim.param_groups[iparam]

                for key in group1:
                    v1 = group1[key]
                    v2 = group2[key]
                    if key != "params":
                        assert v1 == v2


class TestData:
    @pytest.mark.parametrize("mode", ["random", "sequential"])
    def test_split(self, trainer, nequip_dataset, mode):

        trainer.train_val_split = mode
        trainer.set_dataset(nequip_dataset)
        for i, batch in enumerate(trainer.dl_train):
            print(i, batch)


class TestTrain:
    def test_train(self, trainer, nequip_dataset):

        v0 = get_param(trainer.model)
        trainer.set_dataset(nequip_dataset)
        trainer.train()
        v1 = get_param(trainer.model)

        assert not np.allclose(v0, v1), "fail to train parameters"
        assert isfile(trainer.last_model_path), "fail to save best model"

    def test_load_w_revision(self, trainer):

        with tempfile.TemporaryDirectory() as folder:

            file_name = trainer.save(f"{folder}/a.pth")

            dictionary = load_file(
                supported_formats=dict(torch=["pt", "pth"]),
                filename=file_name,
                enforced_format="torch",
            )

            dictionary["root"] = folder
            dictionary["progress"]["stop_arg"] = None
            dictionary["max_epochs"] *= 2
            trainer1 = Trainer.from_dict(dictionary, append=False)
            assert trainer1.iepoch == trainer.iepoch
            assert trainer1.max_epochs == minimal_config["max_epochs"] * 2

    def test_restart_training(self, trainer, nequip_dataset):
        _ = trainer.model
        _ = trainer.device
        _ = trainer.optim
        trainer.set_dataset(nequip_dataset)
        trainer.train()

        v0 = get_param(trainer.model)
        with tempfile.TemporaryDirectory() as folder:
            file_name = trainer.save(f"{folder}/hello.pth")

            dictionary = load_file(
                filename=file_name,
                supported_formats=dict(torch=["pt", "pth"]),
                enforced_format="torch",
            )

            dictionary["progress"]["stop_arg"] = None
            dictionary["max_epochs"] *= 2
            dictionary["root"] = folder

            trainer1 = Trainer.from_dict(dictionary, append=False)
            trainer1.stop_arg = None
            trainer1.max_epochs *= 2
            trainer1.set_dataset(nequip_dataset)
            trainer1.train()
            v1 = get_param(trainer1.model)

            assert not np.allclose(v0, v1), "fail to train parameters"


class TestRescale:
    def test_scaling(self, scale_train):

        trainer = scale_train

        data = trainer.dataset_train[0]

        def get_loss_val(validation):
            trainer.ibatch = 0
            trainer.n_batches = 10
            trainer.reset_metrics()
            trainer.batch_step(
                data=data,
                validation=validation,
            )
            metrics, _ = trainer.metrics.flatten_metrics(
                trainer.batch_metrics,
            )
            return trainer.batch_losses, metrics

        ref_loss, ref_met = get_loss_val(True)
        loss, met = get_loss_val(False)

        # metrics should always be in real unit
        for k in ref_met:
            assert np.allclose(ref_met[k], met[k])
        # loss should always be in normalized unit
        for k in ref_loss:
            assert np.allclose(ref_loss[k], loss[k])

        assert np.allclose(np.sqrt(loss["loss_f"]) * trainer.scale, met["f_rmse"])


def get_param(model):
    v = []
    for p in model.parameters():
        v += [np.hstack(p.data.tolist())]
    v = np.hstack(v)
    return v


class DummyNet(GraphModuleMixin, torch.nn.Module):
    def __init__(self, ndim, nydim=1) -> None:
        super().__init__()
        self.ndim = ndim
        self.nydim = nydim
        self.linear1 = Linear(ndim, nydim)
        self.linear2 = Linear(ndim, nydim * 3)
        self._init_irreps(
            irreps_in={"pos": "1x1o"},
            irreps_out={
                AtomicDataDict.FORCE_KEY: "1x1o",
                AtomicDataDict.TOTAL_ENERGY_KEY: "1x0e",
            },
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = data.copy()
        x = data["pos"]
        data.update(
            {
                AtomicDataDict.FORCE_KEY: self.linear2(x),
                AtomicDataDict.TOTAL_ENERGY_KEY: self.linear1(x),
            }
        )
        return data


class DummyScale(torch.nn.Module):
    """mimic the rescale model"""

    def __init__(self, key, scale, shift) -> None:
        super().__init__()
        self.key = key
        self.scale_by = torch.as_tensor(scale, dtype=torch.get_default_dtype())
        self.shift_by = torch.as_tensor(shift, dtype=torch.get_default_dtype())
        self.linear2 = Linear(3, 3)

    def forward(self, data):
        out = self.linear2(data["pos"])
        if not self.training:
            out = out * self.scale_by
            out = out + self.shift_by
        return {self.key: out}

    def scale(self, data, force_process=False):
        data = data.copy()
        if force_process or not self.training:
            data[self.key] = data[self.key] * self.scale_by
            data[self.key] = data[self.key] + self.shift_by
        return data

    def unscale(self, data, force_process=False):
        data = data.copy()
        if force_process or self.training:
            data[self.key] = data[self.key] - self.shift_by
            data[self.key] = data[self.key] / self.scale_by
        return data


@pytest.fixture(scope="class")
def scale_train(nequip_dataset):
    with tempfile.TemporaryDirectory(prefix="output") as path:
        model = DummyScale(AtomicDataDict.FORCE_KEY, scale=1.3, shift=1)
        config = dict(
            n_train=4,
            n_val=4,
            max_epochs=0,
            batch_size=2,
            loss_coeffs=AtomicDataDict.FORCE_KEY,
            root=path,
            run_name="test_scale",
        )
        trainer = Trainer.from_config(model=model, config=config)
        trainer.set_dataset(nequip_dataset)
        trainer.train()
        trainer.scale = 1.3
        yield trainer
