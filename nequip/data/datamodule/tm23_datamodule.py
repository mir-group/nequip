# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from nequip.data.datamodule import ASEDataModule
from nequip.utils import download_url, extract_zip
from nequip.utils.logger import RankedLogger

import os
from typing import Union, Sequence, List, Callable

logger = RankedLogger(__name__, rank_zero_only=True)

_URL_TM23 = "https://archive.materialscloud.org/record/file?record_id=2113&filename=benchmarking_master_collection-20240316T202423Z-001.zip"
supported_elements = [
    "Ag",
    "Au",
    "Cd",
    "Co",
    "Cr",
    "Cu",
    "Fe",
    "Hf",
    "Hg",
    "Ir",
    "Mn",
    "Mo",
    "Nb",
    "Ni",
    "Os",
    "Pd",
    "Pt",
    "Re",
    "Rh",
    "Ta",
    "Tc",
    "Ti",
    "V",
    "W",
    "Zn",
    "Zr",
]


class TM23DataModule(ASEDataModule):
    """LightningDataModule for the `TM23 dataset <https://www.nature.com/articles/s41524-024-01264-z>`_.

    This datamodule can be used for ``train``, ``validate``, and ``test`` runs.

    This datamodule can automatically download the TM23 dataset from https://archive.materialscloud.org/record/2024.48
    and unzip it in ``data_source_dir`` if not already downloaded. Otherwise, one can download and unzip the dataset as is and
    set ``data_source_dir`` to the directory that contains ``benchmarking_master_collection``.

    The combined dataset containing cold, warm, and melt frames are used as the train and test datasets. ``element`` can be any
    TM23 element, including ``Ag``, ``Au``, ``Cd``, ``Co``, ``Cr``, ``Cu``, ``Fe``, ``Hf``, ``Hg``, ``Ir``, ``Mn``, ``Mo``,
    ``Nb``, ``Ni``, ``Os``, ``Pd``, ``Pt``, ``Re``, ``Rh``, ``Ta``, ``Tc``, ``Ti``, ``V``, ``W``, ``Zn``, and ``Zr``.

    The ``train_val_split`` argument is required to split the training dataset into separate training and validation datasets.

    Args:
        seed (int): data seed for reproducibility
        data_source_dir (str): directory containing the TM23 dataset if present, else directory where TM23 dataset will be downloaded to
        element(str): element from TM23 dataset to use
        transforms (List[Callable]): list of data transforms
        train_val_split (List[float] or List[int]): train-validation split either in fractions ``[1, 1-f]`` or integers ``[N_train, N_val]``
    """

    def __init__(
        self,
        seed: int,
        data_source_dir: str,
        element: str,
        transforms: List[Callable],
        train_val_split: Sequence[Union[int, float]],
        **kwargs,
    ):
        assert element in supported_elements

        train_file_path = "/".join(
            [
                data_source_dir,
                "benchmarking_master_collection",
                element + "_2700cwm_train.xyz",
            ]
        )
        test_file_path = "/".join(
            [
                data_source_dir,
                "benchmarking_master_collection",
                element + "_2700cwm_test.xyz",
            ]
        )
        super().__init__(
            seed=seed,
            test_file_path=test_file_path,
            split_dataset={
                "file_path": train_file_path,
                "train": train_val_split[0],
                "val": train_val_split[1],
            },
            transforms=transforms,
            ase_args={"format": "extxyz"},
            **kwargs,
        )
        self.element = element
        self.data_source_dir = data_source_dir
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path

    def prepare_data(self):
        """"""
        if not (
            os.path.isfile(self.test_file_path) and os.path.isfile(self.train_file_path)
        ):
            download_path = download_url(_URL_TM23, self.data_source_dir)
            extract_zip(download_path, self.data_source_dir)
        else:
            logger.info(
                f"Using existing data files `{self.train_file_path}` and `{self.test_file_path}`"
            )
