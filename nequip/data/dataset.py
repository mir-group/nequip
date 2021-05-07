"""
Dataset classes that parse array of positions, cells to AtomicData object

This module requre the torch_geometric to catch up with the github main branch from Jan. 18, 2021

"""
import numpy as np
import logging

from os.path import dirname, basename, abspath
from typing import Tuple, Dict, Any, List, Callable, Union, Optional

import torch
from torch_geometric.data import Batch, Dataset, download_url, extract_zip

from nequip.data import AtomicData, AtomicDataDict
from ._util import _TORCH_INTEGER_DTYPES


class AtomicDataset(Dataset):
    """The base class for all NequIP datasets."""

    fixed_fields: List[str]
    root: str

    def statistics(
        self, fields: List[Union[str, Callable]], stride: int = 1, unbiased: bool = True
    ) -> List[tuple]:
        """Compute the statistics of ``fields`` in the dataset.

        If the values at the fields are vectors/multidimensional, they must be of fixed shape and elementwise statistics will be computed.

        Args:
            fields: the names of the fields to compute statistics for.
                Instead of a field name, a callable can also be given that reuturns a quantity to compute the statisics for.

                If a callable is given, it will be called with a (possibly batched) ``Data`` object and must return a sequence of points to add to the set over which the statistics will be computed.

                For example, to compute the overall statistics of the x,y, and z components of a per-node vector ``force`` field:

                    data.statistics([lambda data: data.force.flatten()])

                The above computes the statistics over a set of size 3N, where N is the total number of nodes in the dataset.

        Returns:
            List of statistics. For fields of floating dtype the statistics are the two-tuple (mean, std); for fields of integer dtype the statistics are a one-tuple (bincounts,)
        """
        # TODO: If needed, this can eventually be implimented for general AtomicDataset by computing an online running mean and using Welford's method for a stable running standard deviation: https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
        # That would be needed if we have lazy loading datasets.
        # TODO: When lazy-loading datasets are implimented, how to deal with statistics, sampling, and subsets?
        raise NotImplementedError("not implimented for general AtomicDataset yet")


class AtomicInMemoryDataset(AtomicDataset):
    r"""Base class for all datasets that fit in memory.

    By default, the raw file will be stored at root/raw and the processed torch
    file will be at root/process.

    Subclasses must implement:
     - ``raw_file_names``
     - ``get_data()``

    Subclasses may implement:
     - ``download()`` or ``self.url`` or ``ClassName.URL``

     Args:
        file_name (str, optional): file name of data source. only used in children class
        url (str, optional): url to download data source
        root (str, optional): Root directory where the dataset should be saved. Defaults to current working directory.
        force_fixed_keys (list, optional): keys to move from AtomicData to fixed_fields dictionary
        extra_fixed_fields (dict, optional): extra key that are not stored in data but needed for AtomicData initialization
        include_frames (list, optional): the frames to process with the constructor.
    """

    def __init__(
        self,
        root: str,
        file_name: Optional[str] = None,
        url: Optional[str] = None,
        force_fixed_keys: List[str] = [],
        extra_fixed_fields: Dict[str, Any] = {},
        include_frames: Optional[List[int]] = None,
    ):
        # TO DO, this may be symplified
        # See if a subclass defines some inputs
        self.file_name = (
            getattr(type(self), "FILE_NAME", None) if file_name is None else file_name
        )
        force_fixed_keys = set(force_fixed_keys).union(
            getattr(type(self), "FORCE_FIXED_KEYS", [])
        )
        self.url = getattr(type(self), "URL", url)

        self.force_fixed_keys = force_fixed_keys
        self.extra_fixed_fields = extra_fixed_fields
        self.include_frames = include_frames

        self.data = None
        self.fixed_fields = None

        # !!! don't delete this block.
        # otherwise the inherent children class
        # will ignore the download function here
        class_type = type(self)
        if class_type != AtomicInMemoryDataset:
            if "download" not in self.__class__.__dict__:
                class_type.download = AtomicInMemoryDataset.download
            if "process" not in self.__class__.__dict__:
                class_type.process = AtomicInMemoryDataset.process

        # Initialize the InMemoryDataset, which runs download and process
        # See https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets
        # Then pre-process the data if disk files are not found
        super().__init__(root=root)
        if self.data is None:
            self.data, self.fixed_fields, include_frames = torch.load(
                self.processed_paths[0]
            )
            if not np.all(include_frames == self.include_frames):
                raise ValueError(
                    f"the include_frames is changed. "
                    f"please delete the processed folder and rerun {self.processed_paths[0]}"
                )

    @classmethod
    def from_data_list(cls, data_list: List[AtomicData], **kwargs):
        """Make an ``AtomicInMemoryDataset`` from a list of ``AtomicData`` objects.

        Args:
            data_list (List[AtomicData])
            **kwargs: passed through to the constructor
        Returns:
            The constructed ``AtomicInMemoryDataset``.
        """
        obj = cls(**kwargs)
        obj.get_data = lambda: (data_list,)
        return obj

    def len(self):
        if self.data is None:
            return 0
        return self.data.num_graphs

    @property
    def raw_file_names(self):
        raise NotImplementedError()

    @property
    def processed_file_names(self):
        # TO DO, can be updated to hash all simple terms in extra_fixed_fields
        r_max = self.extra_fixed_fields["r_max"]
        dtype = str(torch.get_default_dtype())
        if dtype.startswith("torch."):
            dtype = dtype[len("torch.") :]
        return [f"{r_max}_{dtype}_data.pt"]

    def get_data(
        self,
    ) -> Union[Tuple[Dict[str, Any], Dict[str, Any]], List[AtomicData]]:
        """Get the data --- called from ``process()``, can assume that ``raw_file_names()`` exist.

        Note that parameters for graph construction such as ``pbc`` and ``r_max`` should be included here as (likely, but not necessarily, fixed) fields.

        Returns:
        A two-tuple of:
            fields: dict
                mapping a field name ('pos', 'cell') to a list-like sequence of tensor-like objects giving that field's value for each example.
            fixed_fields: dict
                mapping field names to their constant values for every example in the dataset.
        Or:
            data_list: List[AtomicData]
        """
        raise NotImplementedError

    def download(self):
        if (not hasattr(self, "url")) or (self.url is None):
            # Don't download, assume present. Later could have FileNotFound if the files don't actually exist
            pass
        else:
            download_path = download_url(self.url, self.raw_dir)
            if download_path.endswith(".zip"):
                extract_zip(download_path, self.raw_dir)

    def process(self):
        data = self.get_data()
        if len(data) == 1:

            # It's a data list
            data_list = data[0]
            if not (self.include_frames is None or data[0] is None):
                data_list = [data_list[i] for i in self.include_frames]
            assert all(isinstance(e, AtomicData) for e in data_list)
            assert all(AtomicDataDict.BATCH_KEY not in e for e in data_list)

            fields, fixed_fields = {}, {}

            # take the force_fixed_keys away from the fields
            for key in self.force_fixed_keys:
                if key in data_list[0]:
                    fixed_fields[key] = data_list[0][key]

            fixed_fields.update(self.extra_fixed_fields)

        elif len(data) == 2:

            # It's fields and fixed_fields
            # Get our data
            fields, fixed_fields = data

            fixed_fields.update(self.extra_fixed_fields)

            # check keys
            all_keys = set(fields.keys()).union(fixed_fields.keys())
            assert len(all_keys) == len(fields) + len(
                fixed_fields
            ), "No overlap in keys between data and fixed fields allowed!"
            assert AtomicDataDict.BATCH_KEY not in all_keys
            # Check bad key combinations, but don't require that this be a graph yet.
            AtomicDataDict.validate_keys(all_keys, graph_required=False)

            # take the force_fixed_keys away from the fields
            for key in self.force_fixed_keys:
                if key in fields:
                    fixed_fields[key] = fields.pop(key)[0]

            # check dimesionality
            num_examples = set([len(a) for a in fields.values()])
            if not len(num_examples) == 1:
                raise ValueError(
                    f"This dataset is invalid: expected all fields to have same length (same number of examples), but they had shapes { {f: v.shape for f, v in fields.items() } }"
                )
            num_examples = next(iter(num_examples))

            include_frames = self.include_frames
            if self.include_frames is None:
                include_frames = list(range(num_examples))

            # Make AtomicData from it:
            if AtomicDataDict.EDGE_INDEX_KEY in all_keys:
                # This is already a graph, just build it
                constructor = AtomicData
            else:
                # do neighborlist from points
                constructor = AtomicData.from_points
                assert "r_max" in all_keys
                assert AtomicDataDict.POSITIONS_KEY in all_keys

            data_list = [
                constructor(**{**{f: v[i] for f, v in fields.items()}, **fixed_fields})
                for i in include_frames
            ]

        else:
            raise ValueError("Invalid return from `self.get_data()`")

        # Batch it for efficient saving
        # This limits an AtomicInMemoryDataset to a maximum of LONG_MAX atoms _overall_, but that is a very big number and any dataset that large is probably not "InMemory" anyway
        data = Batch.from_data_list(data_list, exclude_keys=fixed_fields.keys())
        del data_list
        del fields

        # type conversion
        for key, value in fixed_fields.items():
            if isinstance(value, np.ndarray):
                if np.issubdtype(value.dtype, np.floating):
                    fixed_fields[key] = torch.as_tensor(
                        value, dtype=torch.get_default_dtype()
                    )
                else:
                    fixed_fields[key] = torch.as_tensor(value)
            elif np.issubdtype(type(value), np.floating):
                fixed_fields[key] = torch.as_tensor(
                    value, dtype=torch.get_default_dtype()
                )

        logging.info(f"Loaded data: {data}")

        torch.save((data, fixed_fields, self.include_frames), self.processed_paths[0])

        self.data = data
        self.fixed_fields = fixed_fields

    def get(self, idx):
        out = self.data.get_example(idx)
        # Add back fixed fields
        for f, v in self.fixed_fields.items():
            out[f] = v
        return out

    def statistics(
        self,
        fields: List[Union[str, Callable]],
        stride: int = 1,
        unbiased: bool = True,
        modes: Optional[List[Union[str]]] = None,
    ) -> List[tuple]:
        if self.__indices__ is not None:
            selector = torch.as_tensor(self.__indices__)[::stride]
        else:
            selector = torch.arange(0, self.len(), stride)

        node_selector = torch.as_tensor(
            np.in1d(self.data.batch.numpy(), selector.numpy())
        )
        # the pure PyTorch alternative to ^ is:
        # hack for in1d: https://github.com/pytorch/pytorch/issues/3025#issuecomment-392601780
        # node_selector = (self.data.batch[..., None] == selector).any(-1)
        # but this is unnecessary because no backward is done through statistics

        if modes is not None:
            assert len(modes) == len(fields)

        out = []
        for ifield, field in enumerate(fields):

            if field in self.fixed_fields:
                obj = self.fixed_fields
            else:
                obj = self.data

            if callable(field):
                arr = field(obj)
            else:
                arr = obj[field]
            # Apply selector
            # TODO: this might be quite expensive if the dataset is large.
            # Better to impliment the general running average case in AtomicDataset,
            # and just call super here in AtomicInMemoryDataset?
            #
            # TODO: !!! this is a terrible shape-based hack that needs to be fixed !!!
            if len(self.data.batch) == self.data.num_graphs:
                raise NotImplementedError(
                    "AtomicDataset.statistics cannot currently handle datasets whose number of examples is the same as their number of nodes"
                )
            if obj is self.fixed_fields:
                # arr is fixed, nothing to select.
                pass
            elif len(arr) == self.data.num_graphs:
                # arr is per example (probably)
                arr = arr[selector]
            elif len(arr) == len(self.data.batch):
                # arr is per-node (probably)
                arr = arr[node_selector]
            else:
                raise NotImplementedError(
                    "Statistics of properties that are not per-graph or per-node are not yet implimented"
                )

            ana_mode = None if modes is None else modes[ifield]
            if not isinstance(arr, torch.Tensor):
                if np.issubdtype(arr.dtype, np.floating):
                    arr = torch.as_tensor(arr, dtype=torch.get_default_dtype())
                else:
                    arr = torch.as_tensor(arr)
            if ana_mode is None:
                ana_mode = "count" if arr.dtype in _TORCH_INTEGER_DTYPES else "mean_std"

            if ana_mode == "count":
                uniq, counts = torch.unique(
                    torch.flatten(arr), return_counts=True, sorted=True
                )
                out.append((uniq, counts))
            elif ana_mode == "rms":

                out.append((torch.sqrt(torch.mean(arr * arr)),))

            elif ana_mode == "mean_std":

                mean = torch.mean(arr, dim=0)
                std = torch.std(arr, dim=0, unbiased=unbiased)
                out.append((mean, std))

        return out


# TODO: document fixed field mapped key behavior more clearly
class NpzDataset(AtomicInMemoryDataset):
    """Load data from an npz file.

    To avoid loading unneeded data, keys are ignored by default unless they are in ``key_mapping``, ``npz_keys``, or ``npz_fixed_fields``.

    Args:
        file_name (str): file name of the npz file
        key_mapping (Dict[str, str]): mapping of npz keys to ``AtomicData`` keys
        force_fixed_keys (list): keys in the npz to treat as fixed quantities that don't change across examples. For example: cell, atomic_numbers
    """

    def __init__(
        self,
        root: str,
        key_mapping: Dict[str, str] = {
            "positions": AtomicDataDict.POSITIONS_KEY,
            "energy": AtomicDataDict.TOTAL_ENERGY_KEY,
            "force": AtomicDataDict.FORCE_KEY,
            "forces": AtomicDataDict.FORCE_KEY,
            "Z": AtomicDataDict.ATOMIC_NUMBERS_KEY,
            "atomic_number": AtomicDataDict.ATOMIC_NUMBERS_KEY,
        },
        npz_keys: List[str] = [],
        npz_fixed_field_keys: List[str] = [],
        file_name: Optional[str] = None,
        url: Optional[str] = None,
        force_fixed_keys: List[str] = [],
        extra_fixed_fields: Dict[str, Any] = {},
        include_frames: Optional[List[int]] = None,
    ):
        self.key_mapping = key_mapping
        self.npz_fixed_field_keys = npz_fixed_field_keys
        self.npz_keys = npz_keys

        super().__init__(
            file_name=file_name,
            url=url,
            root=root,
            force_fixed_keys=force_fixed_keys,
            extra_fixed_fields=extra_fixed_fields,
            include_frames=include_frames,
        )

    @property
    def raw_file_names(self):
        return [basename(self.file_name)]

    @property
    def raw_dir(self):
        return dirname(abspath(self.file_name))

    # TODO: fixed fields?
    def get_data(self):
        data = np.load(self.raw_dir + "/" + self.raw_file_names[0], allow_pickle=True)

        keys = set(list(self.key_mapping.keys()))
        keys.update(self.npz_fixed_field_keys)
        keys.update(self.npz_keys)
        keys = keys.intersection(set(list(data.keys())))
        mapped = {self.key_mapping.get(k, k): data[k] for k in keys}
        # TODO: generalize this?
        for intkey in (
            AtomicDataDict.ATOMIC_NUMBERS_KEY,
            AtomicDataDict.SPECIES_INDEX_KEY,
            AtomicDataDict.EDGE_INDEX_KEY,
        ):
            if intkey in mapped:
                mapped[intkey] = mapped[intkey].astype(np.int64)

        fields = {k: v for k, v in mapped.items() if k not in self.npz_fixed_field_keys}
        fixed_fields = {
            k: v for k, v in mapped.items() if k in self.npz_fixed_field_keys
        }
        return fields, fixed_fields


class ASEDataset(AtomicInMemoryDataset):
    """TODO

    r_max and an override PBC can be specified in extra_fixed_fields
    """

    def __init__(
        self,
        root: str,
        ase_args: dict = {},
        file_name: Optional[str] = None,
        url: Optional[str] = None,
        force_fixed_keys: List[str] = [],
        extra_fixed_fields: Dict[str, Any] = {},
        include_frames: Optional[List[int]] = None,
    ):

        self.ase_args = dict(index=":")
        self.ase_args.update(getattr(type(self), "ASE_ARGS", dict()))
        self.ase_args.update(ase_args)

        super().__init__(
            file_name=file_name,
            url=url,
            root=root,
            force_fixed_keys=force_fixed_keys,
            extra_fixed_fields=extra_fixed_fields,
            include_frames=include_frames,
        )

    @classmethod
    def from_atoms(cls, atoms: list, **kwargs):
        """Make an ``ASEDataset`` from a list of ``ase.Atoms`` objects.

        Args:
            atoms (List[ase.Atoms])
            **kwargs: passed through to the constructor
        Returns:
            The constructed ``ASEDataset``.
        """
        # TO DO, this funciton fails. It also needs to be unit tested
        obj = cls(**kwargs)
        obj.get_atoms = lambda: atoms
        return obj

    @property
    def raw_file_names(self):
        return [basename(self.file_name)]

    @property
    def raw_dir(self):
        return dirname(abspath(self.file_name))

    def get_atoms(self):
        from ase.io import read as aseread

        return aseread(self.raw_dir + "/" + self.raw_file_names[0], **self.ase_args)

    def get_data(self):
        # Get our data
        atoms_list = self.get_atoms()
        if self.include_frames is None:
            return (
                [
                    AtomicData.from_ase(atoms=atoms, **self.extra_fixed_fields)
                    for atoms in atoms_list
                ],
            )
        else:
            return (
                [
                    AtomicData.from_ase(atoms=atoms_list[i], **self.extra_fixed_fields)
                    for i in self.include_frames
                ],
            )
