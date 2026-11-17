import pytest

import torch
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset

import lightning
from lightning.fabric.plugins.environments.lightning import find_free_network_port
from lightning.pytorch.callbacks import ModelCheckpoint

from nequip.data import AtomicDataDict
from nequip.data.datamodule import NequIPDataModule


class TestDataset:
    def test_subset(self, dataset):
        Subset(dataset, [1, 3])


class TestDataLoader:
    def test_whole(self, dataset):
        dloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=AtomicDataDict.batched_from_list,
        )

        for i, batch in enumerate(dloader):
            print(i)
            print(batch)

    def test_non_divisor(self, dataset):
        dataset = [dataset[i] for i in range(7)]  # make it odd length
        dl = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=AtomicDataDict.batched_from_list,
        )
        dl_iter = iter(dl)
        for _ in range(3):
            batch = next(dl_iter)
            assert AtomicDataDict.num_frames(batch) == 2
        last_batch = next(dl_iter)
        assert AtomicDataDict.num_frames(last_batch) == 1
        assert last_batch["batch"].max() == 0
        with pytest.raises(StopIteration):
            next(dl_iter)

    def test_subset(self, dataset):
        dloader = DataLoader(
            dataset[:4],
            batch_size=2,
            shuffle=True,
            collate_fn=AtomicDataDict.batched_from_list,
        )
        for i, batch in enumerate(dloader):
            print(i)
            print(batch)

    def test_subset_sampler(self, dataset):
        dloader = DataLoader(
            dataset,
            batch_size=2,
            sampler=SubsetRandomSampler(indices=[0, 1, 2, 3, 4]),
            collate_fn=AtomicDataDict.batched_from_list,
        )
        for i, batch in enumerate(dloader):
            print(i)
            print(batch)


class OrderRecorder(lightning.LightningModule):
    """Records the training data of each epoch, in the order it is seen."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.order = {}

    def training_step(self, batch, batch_idx):
        self.order.setdefault(self.current_epoch, []).extend(batch)
        return self.w.square().sum()

    def validation_step(self, batch, batch_idx):
        pass

    def on_train_end(self):
        # anything else that iterates the training data still gets all of it
        loader = self.trainer.train_dataloader
        assert len(list(loader)) == len(loader)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


@pytest.mark.parametrize("strategy", ["auto", "ddp"])
def test_train_data_order(strategy, tmp_path, monkeypatch):
    """Training data order follows `seed`, and resuming training continues it."""
    if strategy == "ddp":
        # preset what Lightning's DDP launcher would otherwise set (`OMP_NUM_THREADS`
        # along with `torch.set_num_threads`), so that it is all undone after the test
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("OMP_NUM_THREADS", str(torch.get_num_threads()))
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", str(find_free_network_port()))
    # dataset items are their own indices, and batches are lists of them
    collate_fn = {"_target_": "builtins.list", "_partial_": True}
    # when `ModelCheckpoint` saves, with 4 batches per epoch:
    checkpoints = {
        "epoch_end": {"save_on_train_epoch_end": True},
        "val_end": {"save_on_train_epoch_end": False, "every_n_epochs": 2},
        # after each epoch's last batch, before the epoch is complete
        "last_batch": {"every_n_train_steps": 4},
        "mid_epoch": {"every_n_train_steps": 3},
    }

    def train(name, seed, ckpt_path=None, **trainer_kwargs):
        datamodule = NequIPDataModule(
            seed=seed,
            train_dataset={"_target_": "builtins.range", "_args_": [16]},
            val_dataset={"_target_": "builtins.range", "_args_": [2]},
            train_dataloader={
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": 4,
                "shuffle": True,
                "num_workers": 1,
                "collate_fn": collate_fn,
            },
            val_dataloader={
                "_target_": "torch.utils.data.DataLoader",
                "collate_fn": collate_fn,
            },
        )
        trainer = lightning.Trainer(
            accelerator="cpu",
            devices=1,
            strategy=strategy,
            default_root_dir=tmp_path / name,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[
                ModelCheckpoint(tmp_path / name / checkpoint, save_last=True, **kwargs)
                for checkpoint, kwargs in checkpoints.items()
            ],
            **trainer_kwargs,
        )
        module = OrderRecorder()
        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
        return module.order

    try:
        uninterrupted = train("uninterrupted", seed=1, max_epochs=3)
        assert train("other_seed", seed=2, max_epochs=1)[0] != uninterrupted[0]
        train("first", seed=1, max_epochs=2)
        for checkpoint in checkpoints:
            ckpt_path = tmp_path / "first" / checkpoint / "last.ckpt"
            resumed = train(checkpoint, seed=1, max_epochs=3, ckpt_path=ckpt_path)
            # the last checkpoints are from the end of epoch 1, except for "mid_epoch"
            # (after 2 of its 4 batches of 4)
            rest = {1: uninterrupted[1][8:]} if checkpoint == "mid_epoch" else {}
            assert resumed == {**rest, 2: uninterrupted[2]}, checkpoint
        # Lightning's epoch counts go out of step when resuming `max_steps` training:
        train("first_steps", seed=1, max_steps=8)
        ckpt_path = tmp_path / "first_steps" / "epoch_end" / "last.ckpt"
        resumed = train("steps", seed=1, max_steps=12, ckpt_path=ckpt_path)
        assert resumed == {2: uninterrupted[2]}
        # Lightning turns shuffling off to overfit:
        overfit = train("overfit", seed=1, max_epochs=2, overfit_batches=2)
        assert overfit == {0: list(range(8)), 1: list(range(8))}
    finally:
        # other code checks whether a process group is initialized
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
