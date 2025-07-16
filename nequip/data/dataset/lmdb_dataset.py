# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from .. import AtomicDataDict
from .base_datasets import AtomicDataset
import lmdb
import pickle
import copy
from dataclasses import dataclass

from functools import cached_property
from typing import (
    Iterable,
    List,
    Callable,
    Union,
    Dict,
    Any,
    Optional,
    Final,
)

NUM_ATOMS_METADATA_KEY: Final[str] = "num_atoms_per_entry"
NUM_EDGES_METADATA_KEY: Final[str] = "num_edges_per_entry"
NUM_FRAMES_METADATA_KEY: Final[str] = "num_frames"


@dataclass(frozen=True)
class LMDBMetadataSpec:
    """
    Describes one metadata column in the LMDB database.

    - name: Key metadata is stored under in LMDB
    - extractor: Function that extracts metadata from AtomicDataDict
    - reducer: Function to combine metadata from each datapoint file
    - initial: Initial value passed to the reducer function
    """

    name: str
    extractor: Callable[[AtomicDataDict.Type], Any]
    reducer: Callable[[Any, Any], Any]
    initial: Any


# Default metadata specifications for LMDB datasets.
_BASE_METADATA: Final[List[LMDBMetadataSpec]] = [
    LMDBMetadataSpec(
        name=NUM_ATOMS_METADATA_KEY,
        extractor=AtomicDataDict.num_nodes,
        reducer=lambda acc, x: (
            acc.append(x) or acc
        ),  # Quicker than lambda acc, x: acc + [x]
        initial=[],
    ),
    LMDBMetadataSpec(
        name=NUM_EDGES_METADATA_KEY,
        extractor=lambda x: (
            AtomicDataDict.num_edges(x) if AtomicDataDict.EDGE_INDEX_KEY in x else None
        ),
        reducer=lambda acc, x: (
            acc.append(x) or acc
        ),  # Quicker than lambda acc, x: acc + [x]
        initial=[],
    ),
    LMDBMetadataSpec(
        name=NUM_FRAMES_METADATA_KEY,
        extractor=AtomicDataDict.num_frames,
        reducer=lambda acc, x: acc + x,
        initial=0,
    ),
]


class NequIPLMDBDataset(AtomicDataset):
    """:class:`~nequip.data.dataset.AtomicDataset` for `LMDB <https://lmdb.readthedocs.io/en/release/>`_ data.

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
        exclude_keys: list[str] = [],
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
        self._length = self.get_metadata(NUM_FRAMES_METADATA_KEY)
        self.exclude_keys = exclude_keys

        # Fallback to stat()['entries'] if no metadata to be backwards compatible
        if self._length is None:
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
                data_list.append({k:v for k,v in pickle.loads(data).items() if k not in self.exclude_keys})
        return data_list

    @classmethod
    def save_from_iterator(
        cls,
        file_path: str,
        iterator: Iterable[AtomicDataDict.Type],
        map_size: int = 53687091200,  # 50 Gb
        write_frequency: int = 1000,
        extra_metadata: List[LMDBMetadataSpec] = [],
    ) -> None:
        """Uses an iterator of ``AtomicDataDict`` objects to construct an LMDB dataset.

        Args:
            file_path (str): path to save the LMDB data
            iterator (Iterable): iterator of atomic data dicts
            map_size (int): maximum size the database may grow to in bytes (defaults to 50 Gb); note that an exception will be raised if database grows larger than map_size
            write_frequency (int): frequency of writing (defaults to 1000). Larger is faster.
            extra_metadata (List[LMDBMetadataSpec]): optional list of extra metadata specifications - beyond _BASE_METADATA - to be written to the database. Defaults to an empty list.
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

        def _write_metadata(metadata_acc, txn):
            metadata = pickle.dumps(metadata_acc, protocol=-1)
            txn.put(b"__metadata__", metadata)

        # Always write base metadata
        metadata_to_write = _BASE_METADATA + copy.deepcopy(extra_metadata)

        metadata_acc = {
            spec.name: copy.deepcopy(spec.initial) for spec in metadata_to_write
        }
        try:
            txn = db.begin(write=True)
            for idx, data in enumerate(iterator):
                # negative number indicates HIGHEST PROTOCOL
                txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
                # Extract metadata and accumulate it
                for spec in metadata_to_write:
                    extracted = spec.extractor(data)
                    metadata_acc[spec.name] = spec.reducer(
                        metadata_acc[spec.name], extracted
                    )
                # commit at each interval
                if idx % write_frequency == 0:
                    _write_metadata(metadata_acc, txn)
                    txn.commit()
                    txn = db.begin(write=True)
            _write_metadata(metadata_acc, txn)
            txn.commit()
        except BaseException:
            txn.abort()
            raise
        finally:
            db.sync()
            db.close()

    @cached_property
    def _metadata(self) -> Dict[str, Any]:
        """
        Load dataset's "__metadata__" key.
        """
        with self.env.begin() as txn:
            raw = txn.get(b"__metadata__")
        if raw is None:
            return {}  # no metadata

        metadata = pickle.loads(raw)
        return metadata

    def get_metadata(self, attr: str, idx: Optional[Union[int, List[int]]] = None):
        if attr in self._metadata:
            metadata_attr = self._metadata[attr]
            if idx is None:
                return metadata_attr
            if isinstance(idx, list):
                return [metadata_attr[_idx] for _idx in idx]
            return metadata_attr[idx]
        return None
