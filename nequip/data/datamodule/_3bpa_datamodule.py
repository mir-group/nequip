from ._ase_datamodule import ASEDataModule
from typing import Union, Sequence, List, Callable, Optional, Dict


test_set_names = ["300K", "600K", "1200K", "dih_beta120", "dih_beta150", "dih_beta180"]


class NequIP3BPADataModule(ASEDataModule):
    """LightningDataModule for the `3BPA dataset <https://pubs.acs.org/doi/full/10.1021/acs.jctc.1c00647>`_.

    This datamodule can be used for ``train``, ``validate``, and ``test`` runs.

    Download the 3BPA `zipfile <https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.1c00647/suppl_file/ct1c00647_si_002.zip>`_
    and unzip it (auto-download fails for this dataset). Pass the path to the directory with the ``.xyz`` files to
    the argument ``data_source_dir``. Users are strongly advised against tampering with the contents of the ``dataset_3BPA``
    directory produced by unzipping -- the logic of this datamodule assumes specific naming conventions.

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
    in terms of how the config arguments:

    ::

      model:
        type_names: [C, H, N, O]
        per_species_rescale_shifts: [-1029.4889999855063, -13.587222780835477, -1484.9814568572233, -2041.9816003861047]

    Args:
        seed (int): data seed for reproducibility
        transforms (List[Callable]): list of data transforms
        train_val_split (List[float]/List[int]): train-validation split either in fractions ``[1, 1-f]`` or integers ``[N_train, N_val]``
        data_source_dir (str): directory containing the 3BPA dataset
        train_set (str): either ``300K`` or ``mixedT``
        test_set (List[str]): list that can contain ``300K``, ``600K``, ``1200K``, ``dih_beta120``, ``dih_beta150``, and/or ``dih_beta180``
        train_dataloader_kwargs (Dict): arguments of the training ``DataLoader``
        val_dataloader_kwargs (Dict): arguments of the validation ``DataLoader``
        test_dataloader_kwargs (Dict): arguments of the testing ``DataLoader``
        stats_manager (Dict): dictionary that can be instantiated into a ``nequip.data.DataStatisticsManager`` object
    """

    def __init__(
        self,
        seed: int,
        transforms: List[Callable],
        train_val_split: Sequence[Union[int, float]],
        data_source_dir: str,
        train_set: str = "300K",
        test_sets: List[str] = test_set_names,
        train_dataloader_kwargs: Dict = {},
        val_dataloader_kwargs: Dict = {},
        test_dataloader_kwargs: Dict = {},
        stats_manager: Optional[Dict] = None,
    ):
        # sanity check
        assert train_set in ["300K" or "mixedT"]
        assert all([tset in test_set_names for tset in test_sets])

        super().__init__(
            seed=seed,
            split_dataset={
                "file_path": data_source_dir + "/train_" + train_set + ".xyz",
                "train": train_val_split[0],
                "val": train_val_split[1],
            },
            test_file_path=[
                data_source_dir + "/test_" + tset + ".xyz" for tset in test_sets
            ],
            transforms=transforms,
            train_dataloader_kwargs=train_dataloader_kwargs,
            val_dataloader_kwargs=val_dataloader_kwargs,
            test_dataloader_kwargs=test_dataloader_kwargs,
            stats_manager=stats_manager,
        )
