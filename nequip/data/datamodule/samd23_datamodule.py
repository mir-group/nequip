from ._ase_datamodule import ASEDataModule
from nequip.utils.file_utils import extract_tar
from nequip.utils.logger import RankedLogger

import os
from typing import List, Callable

logger = RankedLogger(__name__, rank_zero_only=True)

_URLS = {
    "HfO": "https://drive.google.com/uc?id=1-DVMGyXjvNYaBtaAkWu8uQVgvz8pEgMZ",
    "SiN": "https://drive.google.com/uc?id=1l9nsie40Bpm8CNW4sx94yAuvmMkUfM3b",
}


class SAMD23DataModule(ASEDataModule):
    """LightningDataModule for the `Samsung SAMD23 dataset <https://proceedings.neurips.cc/paper_files/paper/2023/hash/a1859debfb3b59d094f3504d5ebb6c25-Abstract-Datasets_and_Benchmarks.html>`_.

    This datamodule can be used for ``train``, ``validate``, and ``test`` runs.

    It automatically downloads the dataset from Google Drive using ``gdown``, extracts it into ``data_source_dir``,
    and loads ASE-compatible datasets from the pre-split ``Trainset.xyz``, ``Validset.xyz``, and ``Testset.xyz`` files.

    If ``include_ood=True``, the datamodule also looks for an ``OOD.xyz`` file in the same folder.
    If found, this file is included as a second test set during evaluation. ``Testset.xyz`` remains the main in-distribution test set.
    This setting does not affect training or validation — only test evaluation.

    Users may also download and extract the data manually.
    In that case, the extracted folder (``HfO/`` or ``SiN/``) should be placed inside ``data_source_dir``, and the expected filenames must be preserved.

    .. note::
        Automatic downloading requires the optional ``gdown`` package. Install with ``pip install gdown``.

    Args:
        seed (int): data seed for reproducibility
        transforms (List[Callable]): list of NequIP data transforms to apply
        data_source_dir (str): directory to store and/or locate the dataset
        system (str): ``HfO`` or ``SiN`` (default ``HfO``)
        include_ood (bool): whether to include ``OOD.xyz`` as a second test set. If True, the test split will contain both ``Testset.xyz`` and ``OOD.xyz``, evaluated as separate test sets. (default ``True``)
    """

    def __init__(
        self,
        seed: int,
        transforms: List[Callable],
        data_source_dir: str,
        system: str = "HfO",
        include_ood: bool = True,
        **kwargs,
    ):
        system = system.strip()
        assert (
            system in _URLS
        ), f"Unknown system `{system}`; must be one of {list(_URLS)}"

        self.system = system
        self.data_source_dir = data_source_dir
        self.dataset_dir = os.path.join(data_source_dir, system)
        self.include_ood = include_ood
        self.ood_path = os.path.join(self.dataset_dir, "OOD.xyz")

        self.train_file_path = os.path.join(self.dataset_dir, "Trainset.xyz")
        self.val_file_path = os.path.join(self.dataset_dir, "Validset.xyz")

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
        """"""
        required_files = [
            self.train_file_path,
            self.val_file_path,
            os.path.join(self.dataset_dir, "Testset.xyz"),
        ]

        if not all(os.path.isfile(f) for f in required_files):
            logger.info(
                f"Dataset files for {self.system} not found locally. Downloading from Google Drive..."
            )

            archive_path = os.path.join(self.data_source_dir, f"{self.system}.tar")

            if not os.path.isfile(archive_path):
                drive_url = _URLS[self.system]
                logger.info(f"Downloading {self.system} dataset from: {drive_url}")

                try:
                    import gdown
                except ImportError as e:
                    raise ImportError(
                        "Downloading the SAMD23 dataset requires the optional 'gdown' package. "
                        "Please install it with `pip install gdown` and try again."
                    ) from e

                gdown.download(drive_url, archive_path, quiet=False)
            else:
                logger.info(f"Archive already exists at: {archive_path}")

            extract_tar(path=archive_path, folder=self.data_source_dir, mode="r:")
        else:
            logger.info(f"Using existing data files in `{self.dataset_dir}`")

        # Log OOD file status after extraction
        if self.include_ood:
            if os.path.isfile(self.ood_path):
                logger.info(f"Confirmed OOD test set exists at: {self.ood_path}")
            else:
                logger.warning(
                    f"OOD test set requested but not found at: {self.ood_path}"
                )
