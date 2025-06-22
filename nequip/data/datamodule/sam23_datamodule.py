from ._ase_datamodule import ASEDataModule
from nequip.utils.file_utils import extract_tar
from nequip.utils.logger import RankedLogger

import os
from typing import Union, List, Callable
import gdown

logger = RankedLogger(__name__, rank_zero_only=True)

_URLS = {
    "HfO": "https://drive.google.com/uc?id=1-DVMGyXjvNYaBtaAkWu8uQVgvz8pEgMZ",
    "SiN": "https://drive.google.com/uc?id=1l9nsie40Bpm8CNW4sx94yAuvmMkUfM3b"
}


class SAM23DataModule(ASEDataModule):
    """Specialized DataModule for the SAM23 HfO and SiN datasets from NeurIPS 2023.

    This module automatically downloads and extracts the dataset,
    then builds ASE-compatible datasets using the pre-split train/val/test files,
    with optional support for an OOD test set.

    Args:
        seed (int): data seed for reproducibility
        transforms (List[Callable]): list of data transforms
        data_source_dir (str): root directory to store the dataset
        data_system (str): "HfO" or "SiN"
        include_ood (bool): whether to include OOD.xyz in test set
    """

    def __init__(
        self,
        seed: int,
        transforms: List[Callable],
        data_source_dir: str,
        data_system: str = "HfO",
        include_ood: bool = False,
        **kwargs,
    ):
        data_system = data_system.strip()
        assert data_system in _URLS, f"Unknown system `{data_system}`; must be one of {list(_URLS)}"

        self.system = data_system
        self.data_source_dir = data_source_dir
        self.dataset_dir = os.path.join(data_source_dir, data_system)
        self.include_ood = include_ood
        self.ood_path = os.path.join(self.dataset_dir, "OOD.xyz")

        self.train_file_path = os.path.join(self.dataset_dir, "Trainset.xyz")
        self.val_file_path   = os.path.join(self.dataset_dir, "Validset.xyz")

        # Always pass both testset and (potentially missing) OOD path to ASEDataModule
        test_file_paths = [os.path.join(self.dataset_dir, "Testset.xyz")]
        if include_ood:
            test_file_paths.append(self.ood_path)

        super().__init__(
            seed=seed,
            train_file_path=self.train_file_path,
            val_file_path=self.val_file_path,
            test_file_path=test_file_paths,
            transforms=transforms,
            **kwargs,
        )

    def prepare_data(self):
        logger.info("[SAM23DataModule] Running prepare_data()")

        required_files = [
            self.train_file_path,
            self.val_file_path,
            os.path.join(self.dataset_dir, "Testset.xyz"),
        ]

        if not all(os.path.isfile(f) for f in required_files):
            logger.info(f"[SAM23DataModule] Dataset files for {self.system} not found locally. Downloading from Google Drive...")

            archive_path = os.path.join(self.data_source_dir, f"{self.system}.tar")

            if not os.path.isfile(archive_path):
                drive_url = _URLS[self.system]
                logger.info(f"[SAM23DataModule] Downloading {self.system} dataset from: {drive_url}")
                gdown.download(drive_url, archive_path, quiet=False)
            else:
                logger.info(f"[SAM23DataModule] Archive already exists at: {archive_path}")

            try:
                extract_tar(path=archive_path, folder=self.data_source_dir, mode="r:")
            except Exception as e:
                logger.error(f"[SAM23DataModule] Extraction failed: {e}")
                raise
        else:
            logger.info(f"[SAM23DataModule] Using existing data files in `{self.dataset_dir}`")

        # Log OOD file status after extraction
        if self.include_ood:
            if os.path.isfile(self.ood_path):
                logger.info(f"[SAM23DataModule] Confirmed OOD test set exists at: {self.ood_path}")
            else:
                logger.warning(f"[SAM23DataModule] OOD test set requested but not found at: {self.ood_path}")
