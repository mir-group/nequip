from nequip.data import AtomicDataDict
from nequip.data.datamodule import ASEDataModule
from nequip.utils.file_utils import download_url
from nequip.utils.logger import RankedLogger

import os
from typing import Union, Sequence, List, Callable

logger = RankedLogger(__name__, rank_zero_only=True)


_URL_WATER = "https://github.com/BingqingCheng/Mapping-the-space-of-materials-and-molecules/raw/refs/heads/master/mlp-water/dataset_1593_eVAng.xyz"


# Least square solve for per-atom energies yields
#    H: -187.6044
#    O: -93.8022


class WaterDataModule(ASEDataModule):
    """LightningDataModule for the water dataset from `Cheng, Bingqing, et al. "Ab initio thermodynamics of liquid and solid water." Proceedings of the National Academy of Sciences 116.4 (2019): 1110-1115. <https://www.pnas.org/doi/full/10.1073/pnas.1815117116>`_.

    Args:
        seed (int): data seed for reproducibility
        transforms (List[Callable]): list of data transforms
        data_source_dir (str): directory that contains ``dataset_1593_eVAng.xyz`` if present, else the directory that ``dataset_1593_eVAng.xyz`` will be downloaded to
        train_val_test_split (Sequence[Union[int, float]]): ``[train, val, test]`` split ratio
    """

    def __init__(
        self,
        seed: int,
        transforms: List[Callable],
        data_source_dir: str,
        train_val_test_split: Sequence[Union[int, float]],
        **kwargs,
    ):
        assert len(train_val_test_split) == 3
        file_path = data_source_dir + "/dataset_1593_eVAng.xyz"

        super().__init__(
            seed=seed,
            split_dataset={
                "file_path": file_path,
                "train": train_val_test_split[0],
                "val": train_val_test_split[1],
                "test": train_val_test_split[2],
            },
            transforms=transforms,
            key_mapping={
                "TotEnergy": AtomicDataDict.TOTAL_ENERGY_KEY,
                "force": AtomicDataDict.FORCE_KEY,
            },
            **kwargs,
        )
        self.data_source_dir = data_source_dir
        self.file_path = file_path

    def prepare_data(self):
        """"""
        if not os.path.isfile(self.file_path):
            _ = download_url(_URL_WATER, self.data_source_dir)
        else:
            logger.info(f"Using existing data file `{self.file_path}`")
