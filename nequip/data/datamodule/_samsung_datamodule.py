# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from ._ase_datamodule import ASEDataModule
from nequip.utils.file_utils import download_url, extract_tar
from nequip.utils.logger import RankedLogger

import os
from typing import Union, List, Callable
import gdown

logger = RankedLogger(__name__, rank_zero_only=True)

_URLS = {
    "HfO": "https://drive.google.com/uc?id=1-DVMGyXjvNYaBtaAkWu8uQVgvz8pEgMZ",
    "SiN": "https://drive.google.com/uc?id=1l9nsie40Bpm8CNW4sx94yAuvmMkUfM3b"
}


class SamsungDataModule(ASEDataModule):
    """Specialized DataModule for the Samsung HfO and SiN datasets from NeurIPS 2023.

    This module automatically downloads and extracts the dataset,
    then builds ASE-compatible datasets using the pre-split train/val/test files.

    Args:
        seed (int): data seed for reproducibility
        transforms (List[Callable]): list of data transforms
        data_source_dir (str): root directory to store the dataset
        dataset_type (str): "HfO" or "SiN"
    """

    def __init__(
        self,
        seed: int,
        transforms: List[Callable],
        data_source_dir: str,
        dataset_type: str = "HfO",
        **kwargs,
    ):
        dataset_type = dataset_type.strip()
        assert dataset_type in _URLS, f"Unknown dataset_type `{dataset_type}`; must be one of {list(_URLS)}"

        self.dataset_type = dataset_type
        self.data_source_dir = data_source_dir
        self.dataset_dir = os.path.join(data_source_dir, f"{dataset_type}")

        self.train_file_path = os.path.join(self.dataset_dir, "Trainset.xyz")
        self.val_file_path = os.path.join(self.dataset_dir, "Validset.xyz")
        self.test_file_path = os.path.join(self.dataset_dir, "Testset.xyz")

        super().__init__(
            seed=seed,
            train_file_path=self.train_file_path,
            val_file_path=self.val_file_path,
            test_file_path=self.test_file_path,
            transforms=transforms,
            **kwargs,
        )

    def prepare_data(self):
        logger.info("[SamsungDataModule] Running prepare_data()")

        expected_files = [
            self.train_file_path,
            self.val_file_path,
            self.test_file_path,
        ]

        if not all(os.path.isfile(f) for f in expected_files):
            logger.info(f"[SamsungDataModule] Dataset files for {self.dataset_type} not found locally. Downloading from Google Drive...")

            # Build target path
            archive_path = os.path.join(self.data_source_dir, f"{self.dataset_type}.tar")

            # Use gdown to download from Google Drive
            if not os.path.isfile(archive_path):
                drive_url = _URLS[self.dataset_type]
                logger.info(f"[SamsungDataModule] Downloading {self.dataset_type} dataset from: {drive_url}")
                gdown.download(drive_url, archive_path, quiet=False)
            else:
                logger.info(f"[SamsungDataModule] Archive already exists at: {archive_path}")

            # Extract the tar file
            try:
                extract_tar(path=archive_path, folder=self.data_source_dir, mode="r:")
            except Exception as e:
                logger.error(f"Extraction failed: {e}")
                raise
        else:
            logger.info(f"[SamsungDataModule] Using existing data files in `{self.dataset_dir}`")