# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from .. import AtomicDataDict
from nequip.utils.logger import RankedLogger

import lightning
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate
import copy
from typing import List, Dict, Any, Union, Optional

logger = RankedLogger(__name__, rank_zero_only=True)


class NequIPDataModule(lightning.LightningDataModule):
    """
    Sanity checking is only performed at runtime -- ensure that the correct datasets are provided for the intended runs,
    which can be ``train``, ``val``, ``test``, and/or ``predict``.

    - ``train`` runs require ``train_dataset`` and ``val_dataset``
    - ``val`` runs require ``val_dataset``
    - ``test`` runs require ``test_dataset``
    - ``predict`` runs require ``predict_dataset``

    One can explicitly specify which ``train``, ``val``, ``test``, ``predict`` datasets to use, or randomly split a dataset to be used for any of those tasks with the ``split_dataset`` argument. These options are not mutually exclusive, e.g. if a single ``test_dataset`` is provided, and ``split_dataset`` is used to get another test set, there will now be two test sets (indexed by ``0`` and ``1``) used for testing. If ``test_dataset`` is a list, i.e. multiple test datasets are provided (e.g. if there are ``n`` test sets with indices ``0``, ``1``, ..., ``n - 1``) and multiple ``split_ataset`` is a list that contributes multiple test sets (say ``m`` such test sets are provided). There will be a total of ``m+n`` test sets, with the ones from ``test_dataset`` taking indices ``0``, ``1``, ..., ``n - 1`` and the ones from the ``split_dataset`` taking indices ``n``, ``n+1``, ..., ``n+m-1``.

    Args:
        seed (int): data seed for reproducibility
        train_dataset (Dict/List[Dict]): training dataset
        val_dataset (Dict/List[Dict]): validation dataset(s) (can provide multiple datasets in a list)
        test_dataset (Dict/List[Dict]): test dataset(s) (can provide multiple datasets in a list)
        predict_dataset (Dict/List[Dict]): prediction dataset(s) (can provide multiple datasets in a list)
        split_dataset (Dict/List[Dict]): dictionary with a ``dataset`` key, which defines the dataset and the keys ``train``, ``val``, ``test``, ``predict`` which represent the subsets to split ``dataset`` into and are either ``int`` s that sum up to the size of ``dataset`` or ``float`` s that sum up to 1 (at least 2, but not necessarily all of ``train``, ``val``, ``test``, ``predict`` must be provided if this option is used)
        train_dataloader (Dict): training ``DataLoader`` configuration dictionary
        val_dataloader (Dict): validation ``DataLoader`` configuration dictionary
        test_dataloader (Dict): testing ``DataLoader`` configuration dictionary
        predict_dataloader (Dict): prediction ``DataLoader`` configuration dictionary
        stats_manager (Dict): dictionary that can be instantiated into a ``nequip.data.DataStatisticsManager`` object
    """

    def __init__(
        self,
        seed: int,
        train_dataset: Optional[Union[Dict, List]] = [],
        val_dataset: Optional[Union[Dict, List]] = [],
        test_dataset: Optional[Union[Dict, List]] = [],
        predict_dataset: Optional[Union[Dict, List]] = [],
        split_dataset: Optional[Union[Dict, List]] = [],
        train_dataloader: Dict = {},
        val_dataloader: Dict = {},
        test_dataloader: Dict = {},
        predict_dataloader: Dict = {},
        stats_manager: Optional[Dict] = None,
    ):
        super().__init__()
        # internal logic follows lists in order of train, val, test, predict, split

        # == first convert all dataset configs to lists if not already lists ==
        dconfigs = []
        for dconfig in [
            train_dataset,
            val_dataset,
            test_dataset,
            predict_dataset,
            split_dataset,
        ]:
            # convert to primitives as later logic is based on types
            if isinstance(dconfig, DictConfig) or isinstance(dconfig, ListConfig):
                dconfig = OmegaConf.to_container(dconfig, resolve=True)
            assert isinstance(dconfig, dict) or isinstance(dconfig, list)
            # make deep copies of the dicts to avoid mutating them in case they are used outside (should be relatively cheap)
            if not isinstance(dconfig, list):
                dconfigs.append([copy.deepcopy(dconfig)])
            else:
                dconfigs.append(copy.deepcopy(dconfig))

        # == account for split datasets ==
        dataset_type_map = ["train", "val", "test", "predict"]  # index matches dconfig
        # loop over datasets to split
        for split_config in dconfigs[4]:
            split_dict = split_config.copy()
            dataset_to_split = split_dict.pop("dataset")
            assert all(
                [dataset_type in dataset_type_map for dataset_type in split_dict.keys()]
            )
            for dataset_type in split_dict:
                # dataset_type is one of "train", "val", "test", "predict"
                dconfigs[dataset_type_map.index(dataset_type)].append(
                    {
                        "_target_": "nequip.data.dataset.RandomSplitAndIndexDataset",
                        "dataset": dataset_to_split,
                        "split_dict": split_dict,
                        "dataset_key": dataset_type,
                        "seed": seed,
                    }
                )

        # == set dataset configs (which are all List[Dict] by this point) ==
        self.train_dataset_config = dconfigs[0]
        self.val_dataset_config = dconfigs[1]
        self.test_dataset_config = dconfigs[2]
        self.predict_dataset_config = dconfigs[3]

        # == keep track of number of each dataset in order of train, val, test, predict ==
        # for communicating to NequIPLightningModule during runs to create enough MetricsManagers to match the number of dataloaders
        self.num_datasets = {
            "train": len(self.train_dataset_config),
            "val": len(self.val_dataset_config),
            "test": len(self.test_dataset_config),
            "predict": len(self.predict_dataset_config),
        }

        logger.info(
            "Found {} training dataset(s), {} validation dataset(s), {} test dataset(s), and {} predict dataset(s).".format(
                self.num_datasets["train"],
                self.num_datasets["val"],
                self.num_datasets["test"],
                self.num_datasets["predict"],
            )
        )

        # == reproducibility params ==
        self.seed = seed
        # distinguish train generator and generators for other datasets
        # to control reproducibility of training runs
        # generators for val/test/predict aren't important since we're only interested in their accumulated metrics
        self.train_generator_state = (
            torch.Generator().manual_seed(self.seed).get_state()
        )
        self.generator_state = torch.Generator().manual_seed(self.seed).get_state()

        # == dataloader ==
        for dataloader_dict, varname in zip(
            [train_dataloader, val_dataloader, test_dataloader, predict_dataloader],
            ["train", "val", "test", "predict"],
        ):
            # copy to be safe against mutation
            dataloader_dict = dataloader_dict.copy()
            if isinstance(dataloader_dict, DictConfig):
                dataloader_dict = OmegaConf.to_container(
                    dataloader_dict.copy(), resolve=True
                )
            assert "dataset" not in dataloader_dict
            assert "generator" not in dataloader_dict
            if "collate_fn" not in dataloader_dict:
                # Allow collate_fn to be overridden by a function wrapping
                # AtomicDataDict.batched_from_list, but default to it.
                dataloader_dict["collate_fn"] = {
                    "_target_": "nequip.data.datamodule._base_datamodule._default_collate_fn_factory"
                }
            setattr(self, f"{varname}_dataloader_config", dataloader_dict)

        # == data statistics manager ==
        self.stats_manager_cfg = stats_manager

    def load_state_dict(self, state_dict: Dict[str, any]) -> None:
        """"""
        self.train_generator_state = state_dict["train_generator_state"]
        self.generator_state = state_dict["generator_state"]

    def state_dict(self) -> Dict[str, Any]:
        """"""
        if self.train_generator is not None:
            train_generator_state = self.train_generator.get_state()
        else:
            train_generator_state = self.train_generator_state
        if self.generator is not None:
            generator_state = self.generator.get_state()
        else:
            generator_state = self.generator_state
        return {
            "train_generator_state": train_generator_state,
            "generator_state": generator_state,
        }

    def setup(self, stage: str) -> None:
        """"""
        self.generator = torch.Generator().manual_seed(self.seed)
        self.generator.set_state(self.generator_state)

        if stage == "fit":
            # requires both "train" and "val" datasets
            if len(self.train_dataset_config) == 0:
                raise RuntimeError("No train dataset provided -- unable to do training")
            else:
                self.train_dataset = instantiate(self.train_dataset_config)
            if len(self.val_dataset_config) == 0:
                raise RuntimeError("No val dataset provided -- unable to do training")
            else:
                self.val_dataset = instantiate(self.val_dataset_config)
            # set train generator
            self.train_generator = torch.Generator().manual_seed(self.seed)
            self.train_generator.set_state(self.train_generator_state)
        elif stage == "validate":
            if len(self.val_dataset_config) == 0:
                raise RuntimeError("No val dataset provided -- unable to do validation")
            else:
                self.val_dataset = instantiate(self.val_dataset_config)
        elif stage == "test":
            if len(self.test_dataset_config) == 0:
                raise RuntimeError("No test dataset provided -- unable to do testing")
            else:
                self.test_dataset = instantiate(self.test_dataset_config)
        elif stage == "predict":
            if len(self.predict_dataset_config) == 0:
                raise RuntimeError("No predict dataset provided -- unable to predict")
            else:
                self.predict_dataset = instantiate(self.predict_dataset_config)
            return

    def teardown(self, stage: str):
        """"""
        self.generator_state = self.generator.get_state()
        del self.generator
        if stage == "fit":
            del self.train_dataset
            del self.val_dataset
            self.train_generator_state = self.train_generator.get_state()
            del self.train_generator
        elif stage == "validate":
            del self.val_dataset
        elif stage == "test":
            del self.test_dataset
        elif stage == "predict":
            del self.predict_dataset

    def train_dataloader(self):
        """"""
        # must only return single train dataloader for now
        # see https://lightning.ai/docs/pytorch/stable/data/iterables.html#multiple-dataloaders
        return self._get_dloader(
            self.train_dataset, self.train_generator, self.train_dataloader_config
        )[0]

    def val_dataloader(self):
        """"""
        return self._get_dloader(
            self.val_dataset, self.generator, self.val_dataloader_config
        )

    def test_dataloader(self):
        """"""
        return self._get_dloader(
            self.test_dataset, self.generator, self.test_dataloader_config
        )

    def predict_dataloader(self):
        """"""
        self._get_dloader(
            self.predict_dataset, self.generator, self.predict_dataloader_config
        )

    def _get_dloader(self, datasets, generator, dataloader_dict):
        return [
            instantiate(
                dataloader_dict,
                dataset=dataset,
                generator=generator,
            )
            for dataset in datasets
        ]

    def get_statistics(self, dataset: str = "train", dataset_idx: int = 0):
        """
        Compute statistics of the dataset.

        Args:
            dataset (str)    : ``train``, ``val``, ``test``, or ``predict``
            dataset_idx (int): dataset index (there can be multiple ``val``, ``test``, ``predict`` datasets)
        """
        if self.stats_manager_cfg is None:
            return {}
        stats_manager = instantiate(self.stats_manager_cfg)  # , _recursive_=False)
        assert dataset in ["train", "val", "test", "predict"]
        task_map = {
            "train": "fit",
            "val": "validate",
            "test": "test",
            "predict": "predict",
        }
        try:
            self.prepare_data()
            self.setup(stage=task_map[dataset])

            # get dataloader, using dataloader_kwargs from the appropriate dataset.
            # stats manager can override options if it wants to, like batch size.
            dataloader_dict = getattr(self, dataset + "_dataloader_config").copy()
            if stats_manager.dataloader_kwargs is not None:
                dataloader_dict.update(stats_manager.dataloader_kwargs)
            dloader = self._get_dloader(
                getattr(self, dataset + "_dataset"), self.generator, dataloader_dict
            )
            dloader = dloader[dataset_idx]

            stats_dict = stats_manager.get_statistics(dloader)
        finally:
            self.teardown(stage=task_map[dataset])
        return stats_dict


def _default_collate_fn_factory() -> callable:
    """Allow `instantiate` to get the default collate_fn by calling this function."""
    return AtomicDataDict.batched_from_list
