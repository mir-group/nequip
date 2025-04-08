# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from ._base_datamodule import NequIPDataModule
from nequip.utils import download_url, extract_zip, extract_tar
from nequip.utils.logger import RankedLogger
from nequip.data import AtomicDataDict

import os
from typing import Union, Sequence, Optional, List, Callable

logger = RankedLogger(__name__, rank_zero_only=True)

KCALMOL_TO_EV = 0.0433641


def _kcalmol_to_ev(data: AtomicDataDict.Type) -> AtomicDataDict.Type:
    data[AtomicDataDict.TOTAL_ENERGY_KEY] = (
        data[AtomicDataDict.TOTAL_ENERGY_KEY] * KCALMOL_TO_EV
    )
    data[AtomicDataDict.FORCE_KEY] = data[AtomicDataDict.FORCE_KEY] * KCALMOL_TO_EV
    return data


class rMD17DataModule(NequIPDataModule):
    """Lightning Data Module responsible for processing rMD17 datasets (including downloading).

    The revised MD-17 datasets can be found at this `link <https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038>`_ .
    This class handles all datasets included in the file: ``aspirin``, ``azobenzene``,  ``benzene``, ``ethanol``, ``malonaldehyde``, ``naphthalene``,
    ``paracetamol``, ``salicylic``, ``toluene`` and ``uracil``. Each dataset contains 100,000 samples for each molecule, with the exception of ``azobenzene`` that contains 99,988 samples.
    Each dataset is not pre-split into training, validation and testing sets. The user has to specify the split using the ``train_val_test_split`` argument.

    .. note::

        If only a subset of the dataset is meant to be used (e.g. for testing), the ``subset_len`` argument can be used to specify the number of samples to use. In this case, ``train_val_test_split`` has to be set either as fractions or as a list of integers that sum up to ``subset_len``. If ``subset_len`` is not set, the full dataset is used.

    Args:
        dataset (str): ``aspirin``, ``azobenzene``,  ``benzene``, ``ethanol``, ``malonaldehyde``, ``naphthalene``, ``paracetamol``, ``salicylic``, ``toluene`` or ``uracil``.
        data_source_dir (str): directory to download the data to, or where the npz files are present if already downloaded and unzipped
        transforms (List[Callable]): list of data transforms
        seed (int): data seed for reproducibility
        train_val_test_split (List[float]/List[int]): train-validation-test split either in fractions ``[a, b, c]`` (``a+b+c=1``) or integers ``[N_train, N_val, N_test]``. If using integers, they have to sum up to either the total number of samples in the dataset, or to the ``subset_len`` if it is set.
        subset_len (int): Subset of ``N_train + N_val + N_test`` to use from the full dataset (the intended use is for minimal tests).
    """

    DATASET_MAP = {
        "aspirin": "rmd17_aspirin.npz",
        "azobenzene": "rmd17_azobenzene.npz",
        "benzene": "rmd17_benzene.npz",
        "ethanol": "rmd17_ethanol.npz",
        "malonaldehyde": "rmd17_malonaldehyde.npz",
        "naphthalene": "rmd17_naphthalene.npz",
        "paracetamol": "rmd17_paracetamol.npz",
        "salicylic": "rmd17_salicylic.npz",
        "toluene": "rmd17_toluene.npz",
        "uracil": "rmd17_uracil.npz",
    }
    DATASET_URL = "https://figshare.com/ndownloader/articles/12672038/versions/3"

    def __init__(
        self,
        dataset: str,
        data_source_dir: str,
        transforms: List[Callable],
        seed: int,
        train_val_test_split: Sequence[Union[int, float]],
        subset_len: Optional[int] = None,
        **kwargs,
    ):

        assert (
            dataset in self.DATASET_MAP.keys()
        ), f"`dataset={dataset}` not supported, `dataset` can be any of {list(self.DATASET_MAP.keys())}"

        file_path = "/".join(
            [data_source_dir, "rmd17/npz_data", self.DATASET_MAP[dataset]]
        )
        # For some reason, `transforms` are loaded as a `omegaconf.ListConfig` and appending
        # the extra function for unit conversion only works when recasting it as a list:
        dataset_config = {
            "_target_": "nequip.data.dataset.NPZDataset",
            "file_path": file_path,
            "transforms": list(transforms) + [_kcalmol_to_ev],
            "key_mapping": {
                "coords": AtomicDataDict.POSITIONS_KEY,
                "nuclear_charges": AtomicDataDict.ATOMIC_NUMBERS_KEY,
                "energies": AtomicDataDict.TOTAL_ENERGY_KEY,
                "forces": AtomicDataDict.FORCE_KEY,
            },
        }

        if subset_len is not None:
            dataset_config = {
                "_target_": "nequip.data.dataset.SubsetByRandomSlice",
                "dataset": dataset_config,
                "start": 0,
                "length": subset_len,
                "seed": seed,
            }

        super().__init__(
            seed=seed,
            split_dataset=[
                {
                    "dataset": dataset_config,
                    "train": train_val_test_split[0],
                    "val": train_val_test_split[1],
                    "test": train_val_test_split[2],
                }
            ],
            **kwargs,
        )
        self.dataset = dataset
        self.data_source_dir = data_source_dir
        self.file_path = file_path

    def prepare_data(self):
        """"""
        if not (os.path.isfile(self.file_path)):
            logger.info(f"Downloading data files to `{self.data_source_dir}`")
            # download and unzip
            download_path = download_url(self.DATASET_URL, self.data_source_dir)
            extract_zip(download_path, self.data_source_dir)
            extract_tar(
                path=self.data_source_dir + "/rmd17.tar.bz2",
                folder=self.data_source_dir,
                mode="r:bz2",
            )

        else:
            logger.info(f"Using existing data files `{self.file_path}`")
