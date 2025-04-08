# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from .. import AtomicDataDict
from ._base_datasets import AtomicDataset
import lmdb
import pickle
from typing import Iterable, List, Callable, Union


class NequIPLMDBDataset(AtomicDataset):
    r"""``AtomicDataset`` for `LMDB <https://lmdb.readthedocs.io/en/release/>`_ data.

    The ``NequIPLMDBDataset`` is the recommended solution for managing large datasets within the NequIP software ecosystem. One can convert existing datasets into LMDB formated data with helper functions from this class.

    As a ``Dataset`` object, this class assumes each entry in the LMDB data is a NequIP ``AtomicDataDict``.

    Args:
        file_path (str): path to LMDB file
        transforms (List[Callable]): list of data transforms
    """

    def __init__(
        self,
        file_path: str,
        transforms: List[Callable] = [],
    ):
        super().__init__(transforms=transforms)
        self.file_path = file_path

        self.env = lmdb.open(
            self.file_path,
            readonly=True,
            lock=False,
            # for better performance on large datasets
            readahead=False,
            subdir=False,
        )

        with self.env.begin() as txn:
            self._length = txn.stat()["entries"]

    def __len__(self):
        return self._length

    def get_data_list(
        self,
        indices: Union[List[int], torch.Tensor, slice],
    ) -> List[AtomicDataDict.Type]:
        if isinstance(indices, slice):
            indices = list(range(*indices.indices(self.num_frames)))

        data_list = []
        with self.env.begin() as txn:
            for idx in indices:
                data = txn.get(f"{idx}".encode("ascii"))
                if data is None:
                    raise IndexError(f"Index {idx} is out of bounds for LMDB dataset.")
                data_list.append(pickle.loads(data))
        return data_list

    @classmethod
    def save_from_iterator(
        self,
        file_path: str,
        iterator: Iterable[AtomicDataDict.Type],
        map_size: int = 53687091200,  # 50 Gb
        write_frequency: int = 1000,
    ) -> None:
        """Uses an iterator of ``AtomicDataDict`` objects to construct an LMDB dataset.

        Args:
            file_path (str): path to save the LMDB data
            iterator (Iterable): iterator of atomic data dicts
            map_size (int): maximum size the database may grow to in bytes (defaults to 50 Gb); note that an exception will be raised if database grows larger than map_size
            write_frequency (int): frequency of writing (defaults to 1000). Larger is faster.
        """
        db = lmdb.open(
            file_path,
            map_size=map_size,
            # not subdirectory but filename prefix
            subdir=False,
            # no zero-initialization of buffers prior to writing them to disk
            meminit=False,
            writemap=True,
            sync=True,
            lock=True,
        )
        try:
            txn = db.begin(write=True)
            for idx, data in enumerate(iterator):
                # negative number indicates HIGHEST PROTOCOL
                txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
                # commit at each interval
                if idx % write_frequency == 0:
                    txn.commit()
                    txn = db.begin(write=True)
            txn.commit()
        except BaseException:
            txn.abort()
            raise
        finally:
            db.sync()
            db.close()
