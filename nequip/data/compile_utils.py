# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from . import AtomicDataDict
from nequip.utils import torch_default_dtype
from nequip.utils.global_dtype import _GLOBAL_DTYPE

import warnings
from hydra.utils import instantiate


def data_dict_from_checkpoint(ckpt_path: str) -> AtomicDataDict.Type:
    with torch_default_dtype(_GLOBAL_DTYPE):
        # === get data from checkpoint ===
        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )
        data_config = checkpoint["hyper_parameters"]["info_dict"]["data"].copy()
        if "train_dataloader" not in data_config:
            data_config["train_dataloader"] = {"_target_: torch.utils.data.DataLoader"}
        data_config["train_dataloader"]["batch_size"] = 1
        datamodule = instantiate(data_config, _recursive_=False)
        # TODO: better way of doing this?
        # instantiate the datamodule, dataset, and get train dataloader
        try:
            datamodule.prepare_data()
            # instantiate train dataset
            datamodule.setup(stage="fit")
            dloader = datamodule.train_dataloader()
            for data in dloader:
                if AtomicDataDict.num_nodes(data) > 3:
                    break
        finally:
            datamodule.teardown(stage="fit")

        # === sanitize data ===
        if AtomicDataDict.CELL_KEY not in data:
            data[AtomicDataDict.CELL_KEY] = 1e5 * torch.eye(
                3,
                dtype=_GLOBAL_DTYPE,
                device=data[AtomicDataDict.POSITIONS_KEY].device,
            ).unsqueeze(0)

            data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = torch.zeros(
                (data[AtomicDataDict.EDGE_INDEX_KEY].size(1), 3),
                dtype=_GLOBAL_DTYPE,
                device=data[AtomicDataDict.POSITIONS_KEY].device,
            )

    return data


def data_dict_from_package(package_path: str) -> AtomicDataDict.Type:
    with warnings.catch_warnings():
        # suppress torch.package TypedStorage warning
        warnings.filterwarnings(
            "ignore",
            message="TypedStorage is deprecated.*",
            category=UserWarning,
            module="torch.package.package_importer",
        )
        imp = torch.package.PackageImporter(package_path)
        data = imp.load_pickle(package="model", resource="example_data.pkl")
    return data
