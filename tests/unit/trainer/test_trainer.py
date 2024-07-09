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
from nequip.nn import GraphModuleMixin, GraphModel, RescaleOutput


def dummy_builder():
    return DummyNet(3)


# set up two config to test
DEBUG = False
NATOMS = 3
NFRAMES = 10
minimal_config = dict(
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
)


def create_trainer(float_tolerance, **kwargs):
    """
    Generate a class instance with minimal configurations,
    with the option to modify the configurations using
    kwargs.
    """
    conf = minimal_config.copy()
    conf.update(kwargs)
    conf["default_dtype"] = str(torch.get_default_dtype())[len("torch.") :]
    model = model_from_config(conf)
    with tempfile.TemporaryDirectory(prefix="output") as path:
        conf["root"] = path
        c = Trainer(model=model, **conf)
        yield c


@pytest.fixture(scope="function")
def trainer(float_tolerance):
    """
    Generate a class instance with minimal configurations.
    """
    yield from create_trainer(float_tolerance)


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
        conf = minimal_config.copy()
        conf["root"] = temp_data

        model = GraphModel(DummyNet(3))
        Trainer(model=model, **conf)

        with pytest.raises(RuntimeError):
            Trainer(model=model, **conf)


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
        for epoch_i in range(3):
            trainer.dl_train_sampler.set_epoch(epoch_i)
            n_samples: int = 0
            for i, batch in enumerate(trainer.dl_train):
                n_samples += batch[AtomicDataDict.BATCH_PTR_KEY].shape[0] - 1
            if trainer.n_train_per_epoch is not None:
                assert n_samples == trainer.n_train_per_epoch
            else:
                assert n_samples == trainer.n_train

    @pytest.mark.parametrize("mode", ["random", "sequential"])
    @pytest.mark.parametrize(
        "n_train_percent, n_val_percent", [("75%", "15%"), ("20%", "30%")]
    )
    def test_split_w_percent_n_train_n_val(
        self, nequip_dataset, mode, float_tolerance, n_train_percent, n_val_percent
    ):
        """
        Test case where n_train and n_val are given as percentage of the
        dataset size, and here they don't sum to 100%.
        """
        # nequip_dataset has 8 frames, so setting n_train to 75% and n_val to 15% should give 6 and 1
        # frames respectively. Note that summed percentages don't have to be 100%
        trainer_w_percent_n_train_n_val = next(
            create_trainer(
                float_tolerance=float_tolerance,
                n_train=n_train_percent,
                n_val=n_val_percent,
            )
        )
        trainer_w_percent_n_train_n_val.train_val_split = mode
        trainer_w_percent_n_train_n_val.set_dataset(nequip_dataset)
        for epoch_i in range(3):
            trainer_w_percent_n_train_n_val.dl_train_sampler.step_epoch(epoch_i)
            n_samples: int = 0
            n_val_samples: int = 0
            for i, batch in enumerate(trainer_w_percent_n_train_n_val.dl_train):
                n_samples += batch[AtomicDataDict.BATCH_PTR_KEY].shape[0] - 1
            if trainer_w_percent_n_train_n_val.n_train_per_epoch is not None:
                assert n_samples == trainer_w_percent_n_train_n_val.n_train_per_epoch
            else:
                assert (
                    n_samples != trainer_w_percent_n_train_n_val.n_train
                )  # n_train now a percentage
                assert trainer_w_percent_n_train_n_val.n_train == n_train_percent  # 75%
                assert n_samples == int(
                    (float(n_train_percent.strip("%")) / 100) * len(nequip_dataset)
                )  # 6
                assert trainer_w_percent_n_train_n_val.n_val == n_val_percent  # 15%

            for i, batch in enumerate(trainer_w_percent_n_train_n_val.dl_val):
                n_val_samples += batch[AtomicDataDict.BATCH_PTR_KEY].shape[0] - 1

            assert (
                n_val_samples != trainer_w_percent_n_train_n_val.n_val
            )  # n_val now a percentage
            assert trainer_w_percent_n_train_n_val.n_val == n_val_percent  # 15%
            assert n_val_samples == int(
                (float(n_val_percent.strip("%")) / 100) * len(nequip_dataset)
            )  # 1 (floored)

    @pytest.mark.parametrize("mode", ["random", "sequential"])
    @pytest.mark.parametrize(
        "n_train_percent, n_val_percent", [("70%", "30%"), ("55%", "45%")]
    )
    def test_split_w_percent_n_train_n_val_flooring(
        self, nequip_dataset, mode, float_tolerance, n_train_percent, n_val_percent
    ):
        """
        Test case where n_train and n_val are given as percentage of the
        dataset size, summing to 100% but with a split that gives
        non-integer numbers of frames for n_train and n_val.
        (i.e. n_train = 70% = 5.6 frames, n_val = 30% = 2.4 frames,
        so final n_train is 6 and n_val is 2)
        """
        # nequip_dataset has 8 frames, so n_train = 70% = 5.6 frames, n_val = 30% = 2.4 frames,
        # so final n_train is 6 and n_val is 2
        trainer_w_percent_n_train_n_val_flooring = next(
            create_trainer(
                float_tolerance=float_tolerance,
                n_train=n_train_percent,
                n_val=n_val_percent,
            )
        )
        trainer_w_percent_n_train_n_val_flooring.train_val_split = mode
        trainer_w_percent_n_train_n_val_flooring.set_dataset(nequip_dataset)
        for epoch_i in range(3):
            trainer_w_percent_n_train_n_val_flooring.dl_train_sampler.step_epoch(
                epoch_i
            )
            n_samples: int = 0
            n_val_samples: int = 0
            for i, batch in enumerate(
                trainer_w_percent_n_train_n_val_flooring.dl_train
            ):
                n_samples += batch[AtomicDataDict.BATCH_PTR_KEY].shape[0] - 1
            if trainer_w_percent_n_train_n_val_flooring.n_train_per_epoch is not None:
                assert (
                    n_samples
                    == trainer_w_percent_n_train_n_val_flooring.n_train_per_epoch
                )
            else:
                assert (
                    n_samples != trainer_w_percent_n_train_n_val_flooring.n_train
                )  # n_train now a percentage
                assert (
                    trainer_w_percent_n_train_n_val_flooring.n_train == n_train_percent
                )  # 70%
                # _not_ equal to the bare floored value now:
                assert n_samples != int(
                    (float(n_train_percent.strip("%")) / 100) * len(nequip_dataset)
                )  # 5
                assert (
                    n_samples
                    == int(  # equal to floored value plus 1
                        (float(n_train_percent.strip("%")) / 100) * len(nequip_dataset)
                    )
                    + 1
                )  # 6
                assert (
                    trainer_w_percent_n_train_n_val_flooring.n_val == n_val_percent
                )  # 30%

            for i, batch in enumerate(trainer_w_percent_n_train_n_val_flooring.dl_val):
                n_val_samples += batch[AtomicDataDict.BATCH_PTR_KEY].shape[0] - 1

            assert (
                n_val_samples != trainer_w_percent_n_train_n_val_flooring.n_val
            )  # n_val now a percentage
            assert (
                trainer_w_percent_n_train_n_val_flooring.n_val == n_val_percent
            )  # 30%
            assert n_val_samples == int(
                (float(n_val_percent.strip("%")) / 100) * len(nequip_dataset)
            )  # 2 (floored)

            assert n_samples + n_val_samples == len(nequip_dataset)  # 100% coverage


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


# subclass to make sure it gets picked up by GraphModel
class DummyScale(RescaleOutput):
    """mimic the rescale model"""

    def __init__(self, key, scale, shift) -> None:
        torch.nn.Module.__init__(self)  # skip RescaleOutput's __init__
        self.key = key
        self.scale_by = torch.as_tensor(scale, dtype=torch.get_default_dtype())
        self.shift_by = torch.as_tensor(shift, dtype=torch.get_default_dtype())
        self.linear2 = Linear(3, 3)
        self.irreps_in = {}
        self.irreps_out = {key: "3x0e"}
        self.model = None

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
        trainer = Trainer(
            model=GraphModel(DummyScale(AtomicDataDict.FORCE_KEY, scale=1.3, shift=1)),
            seed=9,
            n_train=4,
            n_val=4,
            max_epochs=0,
            batch_size=2,
            loss_coeffs=AtomicDataDict.FORCE_KEY,
            root=path,
            run_name="test_scale",
        )
        trainer.set_dataset(nequip_dataset)
        trainer.train()
        trainer.scale = 1.3
        yield trainer
