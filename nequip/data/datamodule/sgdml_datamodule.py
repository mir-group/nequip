# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from ._base_datamodule import NequIPDataModule
from nequip.utils import download_url, extract_zip
from nequip.utils.logger import RankedLogger

import os
from typing import Union, Sequence, Optional, List, Callable

logger = RankedLogger(__name__, rank_zero_only=True)


class sGDML_CCSD_DataModule(NequIPDataModule):
    """Lightning Data Module responsible for processing sGDML CCSD datasets (including downloading).

    The sGDML datasets can be found at http://www.sgdml.org/#datasets. This class handles the CCSD and CCSD(T) datasets,
    including ``aspirin``, ``benzene``, ``malonaldehyde``, ``toluene``, and ``ethanol``.

    Args:
        dataset (str): ``aspirin``, ``benzene``, ``malonaldehyde``, ``toluene``, or ``ethanol``
        data_source_dir (str): directory to download sGDML CCSD data to, or where the npz files are present if already downloaded and unzipped
        transforms (List[Callable]): list of data transforms
        seed (int): data seed for reproducibility
        train_val_split (List[float]/List[int]): train-validation split either in fractions ``[1, 1-f]`` or integers ``[N_train, N_val]``
        trainval_test_subset (List[int]): Subset of ``[N_train + N_val, N_test]`` to use from the full dataset (the intended use is for minimal tests)
    """

    DATASET_MAP = {
        "aspirin": "aspirin_ccsd",
        "benzene": "benzene_ccsd_t",
        "malonaldehyde": "malonaldehyde_ccsd_t",
        "toluene": "toluene_ccsd_t",
        "ethanol": "ethanol_ccsd_t",
    }
    URL_DICT = {
        "aspirin": "http://www.quantum-machine.org/gdml/data/npz/aspirin_ccsd.zip",
        "benzene": "http://www.quantum-machine.org/gdml/data/npz/benzene_ccsd_t.zip",
        "malonaldehyde": "http://www.quantum-machine.org/gdml/data/npz/malonaldehyde_ccsd_t.zip",
        "toluene": "http://www.quantum-machine.org/gdml/data/npz/toluene_ccsd_t.zip",
        "ethanol": "http://www.quantum-machine.org/gdml/data/npz/ethanol_ccsd_t.zip",
    }

    def __init__(
        self,
        dataset: str,
        data_source_dir: str,
        transforms: List[Callable],
        seed: int,
        train_val_split: Sequence[Union[int, float]],
        trainval_test_subset: Optional[List[int]] = None,
        **kwargs,
    ):

        assert (
            dataset in self.DATASET_MAP.keys()
        ), f"`dataset={dataset}` not supported, `dataset` can be any of {list(self.DATASET_MAP.keys())}"

        train_file_path = "/".join(
            [data_source_dir, self.DATASET_MAP[dataset] + "-train.npz"]
        )
        test_file_path = "/".join(
            [data_source_dir, self.DATASET_MAP[dataset] + "-test.npz"]
        )
        trainval_config = {
            "_target_": "nequip.data.dataset.NPZDataset",
            "file_path": train_file_path,
            "transforms": transforms,
        }
        test_config = trainval_config.copy()
        test_config["file_path"] = test_file_path

        if trainval_test_subset is not None:
            assert len(trainval_test_subset) == 2
            trainval_config = {
                "_target_": "nequip.data.dataset.SubsetByRandomSlice",
                "dataset": trainval_config,
                "start": 0,
                "length": trainval_test_subset[0],
                "seed": seed,
            }
            test_config = {
                "_target_": "nequip.data.dataset.SubsetByRandomSlice",
                "dataset": test_config,
                "start": 0,
                "length": trainval_test_subset[1],
                "seed": seed,
            }

        super().__init__(
            seed=seed,
            test_dataset=[test_config],
            split_dataset=[
                {
                    "dataset": trainval_config,
                    "train": train_val_split[0],
                    "val": train_val_split[1],
                }
            ],
            **kwargs,
        )
        self.dataset = dataset
        self.data_source_dir = data_source_dir
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path

    def prepare_data(self):
        """"""
        if not (
            os.path.isfile(self.train_file_path) and os.path.isfile(self.test_file_path)
        ):
            # download and unzip
            download_path = download_url(
                self.URL_DICT[self.dataset], self.data_source_dir
            )
            if download_path.endswith(".zip"):
                extract_zip(download_path, self.data_source_dir)
        else:
            logger.info(
                f"Using existing data files `{self.train_file_path}` and `{self.test_file_path}`"
            )
