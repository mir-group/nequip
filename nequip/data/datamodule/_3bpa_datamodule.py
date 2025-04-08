# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from ._ase_datamodule import ASEDataModule
from nequip.utils.file_utils import download_url, extract_tar
from nequip.utils.logger import RankedLogger

import os
from typing import Union, Sequence, List, Callable

logger = RankedLogger(__name__, rank_zero_only=True)

_URL_3BPA = "https://github.com/davkovacs/BOTNet-datasets/raw/refs/heads/main/dataset_3BPA.tar.gz"
test_set_names = ["300K", "600K", "1200K", "dih_beta120", "dih_beta150", "dih_beta180"]


class NequIP3BPADataModule(ASEDataModule):
    """LightningDataModule for the `3BPA dataset <https://pubs.acs.org/doi/full/10.1021/acs.jctc.1c00647>`_.

    This datamodule can be used for ``train``, ``validate``, and ``test`` runs.

    This datamodule can automatically download the dataset to ``data_source_dir``.
    Users can also manually download the 3BPA `zipfile <https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.1c00647/suppl_file/ct1c00647_si_002.zip>`_ and unzip it (``data_source_dir`` should then be the directory containing the ``dataset_3BPA`` directory). Users must not tamper with the contents of the ``dataset_3BPA`` directory produced upon unzipping as this datamodule assumes the default filenames in the directory.

    The 3BPA dataset has two possible training sets, one at 300K and one with mixed temperatures.
    The ``300K`` training set is used by default, but users can specify it with the
    ``train_set`` -- either ``300K`` or ``mixedT`` is allowed.

    The ``train_val_split`` argument is required to split the ``train_set`` chosen into separate training and
    validation datasets, as in ``nequip.data.NequIPDataModule``.

    There are several test datasets to choose from, including ``300K``, ``600K``, ``1200K``,
    ``dih_beta120``, ``dih_beta150``, and ``dih_beta180``. All are automatically included in the
    testing dataset in that order by default, but one can override this by providing the
    ``test_set`` argument as a ``List`` test sets. One can provide an empty list to have no test sets.

    It is recommended to set the isolated atom energies in the ``model``'s ``per_species_rescale_shifts``.
    The following information can be found in ``iso_atoms.xyz`` in the 3BPA data zip, but is reproduced here
    in the format of the config arguments:

    ::

      model:
        type_names: [C, H, N, O]
        per_species_rescale_shifts: [-1029.4889999855063, -13.587222780835477, -1484.9814568572233, -2041.9816003861047]

    Args:
        seed (int): data seed for reproducibility
        transforms (List[Callable]): list of data transforms
        train_val_split (List[float]/List[int]): train-validation split either in fractions ``[1, 1-f]`` or integers ``[N_train, N_val]``
        data_source_dir (str): directory to download 3BPA dataset to, or where the ``dataset_3BPA`` directory is located if already downloaded and unzipped
        train_set (str): either ``300K`` or ``mixedT``
        test_set (List[str]): list that can contain ``300K``, ``600K``, ``1200K``, ``dih_beta120``, ``dih_beta150``, and/or ``dih_beta180``
    """

    def __init__(
        self,
        seed: int,
        transforms: List[Callable],
        train_val_split: Sequence[Union[int, float]],
        data_source_dir: str,
        train_set: str = "300K",
        test_sets: List[str] = test_set_names,
        **kwargs,
    ):
        # sanity check
        assert train_set in ["300K" or "mixedT"]
        assert all([tset in test_set_names for tset in test_sets])

        train_file_path = data_source_dir + "/dataset_3BPA/train_" + train_set + ".xyz"
        test_file_paths = [
            data_source_dir + "/dataset_3BPA/test_" + tset + ".xyz"
            for tset in test_sets
        ]
        super().__init__(
            seed=seed,
            split_dataset={
                "file_path": train_file_path,
                "train": train_val_split[0],
                "val": train_val_split[1],
            },
            test_file_path=test_file_paths,
            transforms=transforms,
            **kwargs,
        )
        self.data_source_dir = data_source_dir
        self.train_file_path = train_file_path
        self.test_file_paths = test_file_paths

    def prepare_data(self):
        """"""
        if not all(
            [
                os.path.isfile(path)
                for path in self.test_file_paths + [self.train_file_path]
            ]
        ):
            download_path = download_url(_URL_3BPA, self.data_source_dir)
            extract_tar(download_path, self.data_source_dir)
        else:
            logger.info(f"Using existing data files in `{self.data_source_dir}`")
