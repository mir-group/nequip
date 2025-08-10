from nequip.data.datamodule import ASEDataModule
from nequip.utils.file_utils import download_url
from nequip.utils.logger import RankedLogger

import os
from typing import List, Callable

logger = RankedLogger(__name__, rank_zero_only=True)


_URL_TRAIN = "https://figshare.com/ndownloader/files/25605734"
_URL_VAL = "https://figshare.com/ndownloader/files/25605737"
_URL_TEST = "https://figshare.com/ndownloader/files/25605740"

# Least square solve for per-atom energies yields
#    C: -1035.412048 (0.036)
#    H: -16.834627 (0.023)
#    O: -2046.033121 (0.041)


class COLLDataModule(ASEDataModule):
    """LightningDataModule for the COLL dataset from `<https://arxiv.org/abs/2011.14115>`_.

    Args:
        seed (int): data seed for reproducibility
        transforms (List[Callable]): list of data transforms
        data_source_dir (str): directory where dataset files will be downloaded to if not already present
    """

    def __init__(
        self,
        seed: int,
        transforms: List[Callable],
        data_source_dir: str,
        **kwargs,
    ):
        self.data_source_dir = data_source_dir

        train_file_path = os.path.join(data_source_dir, "coll_v1.2_AE_train.xyz")
        val_file_path = os.path.join(data_source_dir, "coll_v1.2_AE_val.xyz")
        test_file_path = os.path.join(data_source_dir, "coll_v1.2_AE_test.xyz")

        super().__init__(
            seed=seed,
            train_file_path=train_file_path,
            val_file_path=val_file_path,
            test_file_path=test_file_path,
            transforms=transforms,
            **kwargs,
        )

        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.test_file_path = test_file_path

    def prepare_data(self):
        """"""
        os.makedirs(self.data_source_dir, exist_ok=True)

        files_to_download = [
            (self.train_file_path, _URL_TRAIN, "training"),
            (self.val_file_path, _URL_VAL, "validation"),
            (self.test_file_path, _URL_TEST, "test"),
        ]

        for file_path, url, dataset_type in files_to_download:
            if not os.path.isfile(file_path):
                logger.info(f"Downloading {dataset_type} dataset to `{file_path}`")
                download_url(
                    url, self.data_source_dir, filename=os.path.basename(file_path)
                )
            else:
                logger.info(f"Using existing {dataset_type} data file `{file_path}`")
