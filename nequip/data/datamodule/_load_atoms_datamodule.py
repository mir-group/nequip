# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from ._base_datamodule import NequIPDataModule
from nequip.utils.logger import RankedLogger

import os
import tempfile
from typing import Union, Sequence, Optional, List, Callable

logger = RankedLogger(__name__, rank_zero_only=True)


class LoadAtomsDataModule(NequIPDataModule):
    """Lightning Data Module for datasets provided by the
    `load-atoms <https://github.com/jla-gardner/load-atoms>`_ package.

    Requires the ``load-atoms`` package: ``pip install load-atoms``.

    Any dataset available via ``load_atoms.load_dataset()`` can be used
    directly without manual downloading or format conversion.

    Args:
        dataset_id (str): identifier of the dataset, e.g. ``"C-GAP-17"``, ``"QM9"``, ``"rMD17-aspirin"``.
        data_source_dir (str): directory where load-atoms will cache the downloaded dataset.
            Defaults to a system temporary directory if not provided.
        transforms (List[Callable]): list of data transforms to apply.
        seed (int): data seed for reproducibility.
        train_val_test_split (List[float] or List[int]): train/val/test split as fractions
            (must sum to 1.0) or as integers (must sum to the dataset size or ``subset_len``).
        subset_len (int, optional): if set, only use this many structures from the dataset.
        include_keys (List[str], optional): additional per-atom or per-structure keys to read.
        exclude_keys (List[str], optional): keys to exclude from the ASE Atoms objects.
        key_mapping (dict, optional): mapping from ASE keys to AtomicDataDict keys.
    """

    def __init__(
        self,
        dataset_id: str,
        transforms: List[Callable],
        seed: int,
        train_val_test_split: Sequence[Union[int, float]],
        data_source_dir: Optional[str] = None,
        subset_len: Optional[int] = None,
        include_keys: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        key_mapping: Optional[dict] = None,
        **kwargs,
    ):
        self.dataset_id = dataset_id
        self.data_source_dir = data_source_dir or tempfile.gettempdir()
        self._extxyz_path = os.path.join(self.data_source_dir, f"{dataset_id}.extxyz")

        dataset_config = {
            "_target_": "nequip.data.dataset.ASEDataset",
            "file_path": self._extxyz_path,
            "transforms": list(transforms),
            "include_keys": include_keys or [],
            "exclude_keys": exclude_keys or [],
            "key_mapping": key_mapping or {},
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

    def prepare_data(self) -> None:
        """Download the dataset via load-atoms and write to extxyz for ASEDataset."""

        if os.path.isfile(self._extxyz_path):
            logger.info(f"Using existing cached dataset at `{self._extxyz_path}`")
            return

        try:
            from load_atoms import load_dataset
            import ase.io
        except ImportError:
            raise ImportError(
                "The `load-atoms` package is required for LoadAtomsDataModule. "
                "Install it with: pip install load-atoms"
            )

        logger.info(
            f"Downloading `{self.dataset_id}` via load-atoms to `{self.data_source_dir}`"
        )
        dataset = load_dataset(self.dataset_id, root=self.data_source_dir)

        logger.info(f"Writing {len(dataset)} structures to `{self._extxyz_path}`")
        ase.io.write(self._extxyz_path, list(dataset))
        logger.info("Done.")
