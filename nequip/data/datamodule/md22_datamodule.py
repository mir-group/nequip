# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from ._base_datamodule import NequIPDataModule
from nequip.data import AtomicDataDict
from nequip.utils import download_url, extract_zip
from nequip.utils.logger import RankedLogger

import os
from typing import Union, Sequence, List, Callable

logger = RankedLogger(__name__, rank_zero_only=True)

KCALMOL_TO_EV = 0.0433641


def _kcalmol_to_ev(data: AtomicDataDict.Type) -> AtomicDataDict.Type:
    data[AtomicDataDict.TOTAL_ENERGY_KEY] = (
        data[AtomicDataDict.TOTAL_ENERGY_KEY] * KCALMOL_TO_EV
    )
    data[AtomicDataDict.FORCE_KEY] = data[AtomicDataDict.FORCE_KEY] * KCALMOL_TO_EV
    return data


class MD22DataModule(NequIPDataModule):
    """Lightning Data Module responsible for processing sGDML MD22 datasets (including downloading).

    This class handles the MD22 datasets, including ``tetrapeptide`` (CHNO), ``dha`` (CHO), ``stachyose`` (CHO), ``dna_atat`` (CHNO), ``dna_atat_cgcg`` (CHNO), ``buckyball_catcher`` (CH), and ``double_walled_nanotube`` (CH).
    See `Science Advances 9.2 (2023): eadf0873 <https://www.science.org/doi/10.1126/sciadv.adf0873>`_ for more details.

    This datamodule will automatically use the training set sizes from the paper, that is, ``tetrapeptide`` (6,000/85,109), ``dha`` (8,000/69,753), ``stachyose`` (8,000/27,272), ``dna_atat`` (3,000/20,001), ``dna_atat_cgcg`` (2,000/10,153), ``buckyball_catcher`` (600/6,102), and ``double_walled_nanotube`` (800/5,032). The "training set" will then be partitioned into train and validation datasets based on ``train_val_split``. The remainder is used as the test dataset.

    Args:
        dataset (str): ``tetrapeptide``, ``dha``, ``stachyose``, ``dna_atat``, ``dna_atat_cgcg``, ``buckyball_catcher``, or ``double_walled_nanotube``
        data_source_dir (str): directory to download sGDML MD22 data to, or where the npz files are present if already downloaded
        transforms (List[Callable]): list of data transforms
        seed (int): data seed for reproducibility
        train_val_split (List[float] or List[int]): train-validation split either in fractions ``[1, 1-f]`` or integers ``[N_train, N_val]``
    """

    # dataset: [file_name, num_trainval, num_data]
    dataset_map = {
        "tetrapeptide": ["md22_Ac-Ala3-NHMe.npz", 6000, 85109],
        "dha": ["md22_DHA.npz", 8000, 69753],
        "stachyose": ["md22_stachyose.npz", 8000, 27272],
        "dna_atat": ["md22_AT-AT.npz", 3000, 20001],
        "dna_atat_cgcg": ["md22_AT-AT-CG-CG.npz", 2000, 10153],
        "buckyball_catcher": ["md22_buckyball-catcher.npz", 600, 6102],
        "double_walled_nanotube": ["md22_double-walled_nanotube.npz", 800, 5032],
    }
    url_dict = {
        "tetrapeptide": "http://www.quantum-machine.org/gdml/repo/datasets/md22_Ac-Ala3-NHMe.npz",
        "dha": "http://www.quantum-machine.org/gdml/repo/datasets/md22_DHA.npz",
        "stachyose": "http://www.quantum-machine.org/gdml/repo/datasets/md22_stachyose.npz",
        "dna_atat": "http://www.quantum-machine.org/gdml/repo/datasets/md22_AT-AT.npz",
        "dna_atat_cgcg": "http://www.quantum-machine.org/gdml/repo/datasets/md22_AT-AT-CG-CG.npz",
        "buckyball_catcher": "http://www.quantum-machine.org/gdml/repo/datasets/md22_buckyball-catcher.npz",
        "double_walled_nanotube": "http://www.quantum-machine.org/gdml/repo/datasets/md22_double-walled_nanotube.npz",
    }

    def __init__(
        self,
        dataset: str,
        data_source_dir: str,
        transforms: List[Callable],
        seed: int,
        train_val_split: Sequence[Union[int, float]],
        **kwargs,
    ):

        assert (
            dataset in self.dataset_map.keys()
        ), f"`dataset={dataset}` not supported, `dataset` can be any of {list(self.dataset_map.keys())}"

        data_file_path = "/".join([data_source_dir, self.dataset_map[dataset][0]])
        base_config = {
            "_target_": "nequip.data.dataset.NPZDataset",
            "file_path": data_file_path,
            "transforms": list(transforms) + [_kcalmol_to_ev],
            "key_mapping": {
                "R": AtomicDataDict.POSITIONS_KEY,
                "z": AtomicDataDict.ATOMIC_NUMBERS_KEY,
                "E": AtomicDataDict.TOTAL_ENERGY_KEY,
                "F": AtomicDataDict.FORCE_KEY,
            },
        }

        train_end_idx = self.dataset_map[dataset][1]
        test_length = self.dataset_map[dataset][2] - train_end_idx

        train_config = {
            "_target_": "nequip.data.dataset.SubsetByRandomSlice",
            "dataset": base_config,
            "start": 0,
            "length": train_end_idx,
            "seed": seed,
        }

        test_config = {
            "_target_": "nequip.data.dataset.SubsetByRandomSlice",
            "dataset": base_config,
            "start": train_end_idx,
            "length": test_length,
            "seed": seed,
        }

        super().__init__(
            seed=seed,
            split_dataset={
                "dataset": train_config,
                "train": train_val_split[0],
                "val": train_val_split[1],
            },
            test_dataset=test_config,
            **kwargs,
        )
        self.dataset = dataset
        self.data_source_dir = data_source_dir
        self.data_file_path = data_file_path

    def prepare_data(self):
        """"""
        if not os.path.isfile(self.data_file_path):
            # download and unzip
            download_path = download_url(
                self.url_dict[self.dataset], self.data_source_dir
            )
            if download_path.endswith(".zip"):
                extract_zip(download_path, self.data_source_dir)
        else:
            logger.info(f"Using existing data file `{self.data_file_path}`")
