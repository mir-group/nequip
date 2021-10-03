import numpy as np
import logging
import tempfile
import inspect
from torch._C import Value
import yaml
import hashlib
from os.path import dirname, basename, abspath
from typing import Tuple, Dict, Any, List, Callable, Union, Optional, Sequence

import ase

import torch

from torch_scatter import scatter, scatter_std
from nequip.utils.torch_geometric import Batch, Dataset
from nequip.utils.torch_geometric.utils import download_url, extract_zip

import nequip
from nequip.data import (
    AtomicData,
    AtomicDataDict,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _GRAPH_FIELDS,
)
from nequip.utils.batch_ops import bincount
from nequip.utils.regressor import gp
from ._util import _TORCH_INTEGER_DTYPES
from .transforms import TypeMapper


class AtomicDataset(Dataset):
    """The base class for all NequIP datasets."""

    fixed_fields: List[str]
    root: str

    def statistics(
        self,
        fields: List[Union[str, Callable]],
        modes: List[str],
        stride: int = 1,
        unbiased: bool = True,
        kwargs: Optional[Dict[str, dict]] = {},
    ) -> List[tuple]:
        # TODO: If needed, this can eventually be implimented for general AtomicDataset by computing an online running mean and using Welford's method for a stable running standard deviation: https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
        # That would be needed if we have lazy loading datasets.
        # TODO: When lazy-loading datasets are implimented, how to deal with statistics, sampling, and subsets?
        raise NotImplementedError("not implimented for general AtomicDataset yet")

    @property
    def type_mapper(self) -> Optional[TypeMapper]:
        # self.transform is always a TypeMapper
        return self.transform


class AtomicInMemoryDataset(AtomicDataset):
    r"""Base class for all datasets that fit in memory.

    Please note that, as a ``pytorch_geometric`` dataset, it must be backed by some kind of disk storage.
    By default, the raw file will be stored at root/raw and the processed torch
    file will be at root/process.

    Subclasses must implement:
     - ``raw_file_names``
     - ``get_data()``

    Subclasses may implement:
     - ``download()`` or ``self.url`` or ``ClassName.URL``

    Args:
        root (str, optional): Root directory where the dataset should be saved. Defaults to current working directory.
        file_name (str, optional): file name of data source. only used in children class
        url (str, optional): url to download data source
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
        type_mapper: TypeMapper = None,
    ):
        # TO DO, this may be simplified
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
        super().__init__(root=root, transform=type_mapper)
        if self.data is None:
            self.data, self.fixed_fields, include_frames = torch.load(
                self.processed_paths[0]
            )
            if not np.all(include_frames == self.include_frames):
                raise ValueError(
                    f"the include_frames is changed. "
                    f"please delete the processed folder and rerun {self.processed_paths[0]}"
                )

    def len(self):
        if self.data is None:
            return 0
        return self.data.num_graphs

    @property
    def raw_file_names(self):
        raise NotImplementedError()

    def _get_parameters(self) -> Dict[str, Any]:
        """Get a dict of the parameters used to build this dataset."""
        pnames = list(inspect.signature(self.__init__).parameters)
        IGNORE_KEYS = {
            # the type mapper is applied after saving, not before, so doesn't matter to cache validity
            "type_mapper"
        }
        params = {
            k: getattr(self, k)
            for k in pnames
            if k not in IGNORE_KEYS and hasattr(self, k)
        }
        # Add other relevant metadata:
        params["dtype"] = str(torch.get_default_dtype())
        params["nequip_version"] = nequip.__version__
        return params

    @property
    def processed_dir(self) -> str:
        # We want the file name to change when the parameters change
        # So, first we get all parameters:
        params = self._get_parameters()
        # Make some kind of string of them:
        # we don't care about this possibly changing between python versions,
        # since a change in python version almost certainly means a change in
        # versions of other things too, and is a good reason to recompute
        buffer = yaml.dump(params).encode("ascii")
        # And hash it:
        param_hash = hashlib.sha1(buffer).hexdigest()
        return f"{self.root}/processed_dataset_{param_hash}"

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pth", "params.yaml"]

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
        with open(self.processed_paths[1], "w") as f:
            yaml.dump(self._get_parameters(), f)

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
        modes: List[str],
        stride: int = 1,
        unbiased: bool = True,
        kwargs: Optional[Dict[str, dict]] = {},
    ) -> List[tuple]:
        """Compute the statistics of ``fields`` in the dataset.

        If the values at the fields are vectors/multidimensional, they must be of fixed shape and elementwise statistics will be computed.

        Args:
            fields: the names of the fields to compute statistics for.
                Instead of a field name, a callable can also be given that reuturns a quantity to compute the statisics for.

                If a callable is given, it will be called with a (possibly batched) ``Data``-like object and must return a sequence of points to add to the set over which the statistics will be computed.
                The callable must also return a string, one of ``"node"`` or ``"graph"``, indicating whether the value it returns is a per-node or per-graph quantity.
                PLEASE NOTE: the argument to the callable may be "batched", and it may not be batched "contiguously": ``batch`` and ``edge_index`` may have "gaps" in their values.

                For example, to compute the overall statistics of the x,y, and z components of a per-node vector ``force`` field:

                    data.statistics([lambda data: (data.force.flatten(), "node")])

                The above computes the statistics over a set of size 3N, where N is the total number of nodes in the dataset.

            modes: the statistic to compute for each field. Valid options are TODO.

            stride: the stride over the dataset while computing statistcs.

            unbiased: whether to use unbiased for standard deviations.

            kwargs: other options for individual statistics modes.

        Returns:
            List of statistics. For fields of floating dtype the statistics are the two-tuple (mean, std); for fields of integer dtype the statistics are a one-tuple (bincounts,)
        """

        # Short circut:
        assert len(modes) == len(fields)
        if len(fields) == 0:
            return []

        if self._indices is not None:
            graph_selector = torch.as_tensor(self._indices)[::stride]
        else:
            graph_selector = torch.arange(0, self.len(), stride)
        num_graphs = len(graph_selector)

        node_selector = torch.as_tensor(
            np.in1d(self.data.batch.numpy(), graph_selector.numpy())
        )
        num_nodes = node_selector.sum()

        if self.transform is not None:
            # pre-transform the fixed fields and data so that statistics process transformed data
            ff_transformed = self.transform(self.fixed_fields, types_required=False)
            data_transformed = self.transform(self.data.to_dict(), types_required=False)
        else:
            ff_transformed = self.fixed_fields
            data_transformed = self.data.to_dict()
        # pre-select arrays
        # this ensures that all following computations use the right data
        selectors = {}
        for k in list(ff_transformed.keys()) + list(data_transformed.keys()):
            if k in _NODE_FIELDS:
                selectors[k] = node_selector
            elif k in _GRAPH_FIELDS:
                selectors[k] = graph_selector
            # TODO: edges?
        # TODO: do the batch indexes, edge_indexes, etc. after selection need to be
        # "compacted" to subtract out their offsets? For now, we just punt this
        # onto the writer of the callable field.
        # do not actually select on fixed fields, since they are constant
        # but still only select fields that are correctly registered
        ff_transformed = {k: v for k, v in ff_transformed.items() if k in selectors}
        # apply selector to actual data
        data_transformed = {
            k: data_transformed[k][selectors[k]]
            for k in data_transformed.keys()
            if k in selectors
        }

        atom_types: Optional[torch.Tensor] = None
        out: list = []
        for ifield, field in enumerate(fields):
            if field in self.fixed_fields:
                obj = ff_transformed
            else:
                obj = data_transformed

            if callable(field):
                arr, arr_is_per = field(obj)
                assert arr_is_per in ("node", "graph")
            else:
                # Give a better error
                if field not in selectors:
                    # this means field is not selected and so not available
                    raise RuntimeError(
                        f"Only per-node and per-graph fields can have statistics computed; `{field}` has not been registered as either. If it is per-node or per-graph, please register it as such using `nequip.data.register_fields`"
                    )
                arr = obj[field]
                if field in _NODE_FIELDS:
                    arr_is_per = "node"
                elif field in _GRAPH_FIELDS:
                    arr_is_per = "graph"
                else:
                    raise RuntimeError

            # Check arr
            if arr is None:
                raise ValueError(
                    f"Cannot compute statistics over field `{field}` whose value is None!"
                )
            if not isinstance(arr, torch.Tensor):
                if np.issubdtype(arr.dtype, np.floating):
                    arr = torch.as_tensor(arr, dtype=torch.get_default_dtype())
                else:
                    arr = torch.as_tensor(arr)
            if arr_is_per == "node":
                arr = arr.view(num_nodes, -1)
            elif arr_is_per == "graph":
                arr = arr.view(num_graphs, -1)

            ana_mode = modes[ifield]
            # compute statistics
            if ana_mode == "count":
                # count integers
                uniq, counts = torch.unique(
                    torch.flatten(arr), return_counts=True, sorted=True
                )
                out.append((uniq, counts))
            elif ana_mode == "rms":
                # root-mean-square
                out.append((torch.sqrt(torch.mean(arr * arr)),))

            elif ana_mode == "mean_std":
                # mean and std
                mean = torch.mean(arr, dim=0)
                std = torch.std(arr, dim=0, unbiased=unbiased)
                out.append((mean, std))

            elif ana_mode.startswith("per_species_"):
                # per-species
                algorithm_kwargs = kwargs.pop(field + ana_mode, {})

                ana_mode = ana_mode[len("per_species_") :]

                if atom_types is None:
                    if AtomicDataDict.ATOM_TYPE_KEY in data_transformed:
                        atom_types = data_transformed[AtomicDataDict.ATOM_TYPE_KEY]
                    elif AtomicDataDict.ATOM_TYPE_KEY in ff_transformed:
                        atom_types = ff_transformed[AtomicDataDict.ATOM_TYPE_KEY]
                        atom_types = (
                            atom_types.unsqueeze(0)
                            .expand((num_graphs,) + atom_types.shape)
                            .reshape(-1)
                        )

                results = self._per_species_statistics(
                    ana_mode,
                    arr,
                    arr_is_per=arr_is_per,
                    batch=data_transformed[AtomicDataDict.BATCH_KEY],
                    atom_types=atom_types,
                    unbiased=unbiased,
                    **algorithm_kwargs,
                )
                out.append(results)

            elif ana_mode.startswith("per_atom_"):
                # per-atom
                # only makes sense for a per-graph quantity
                if arr_is_per != "graph":
                    raise ValueError(
                        f"It doesn't make sense to ask for `{ana_mode}` since `{field}` is not per-graph"
                    )
                ana_mode = ana_mode[len("per_atom_") :]
                results = self._per_atom_statistics(
                    ana_mode=ana_mode,
                    arr=arr,
                    batch=data_transformed[AtomicDataDict.BATCH_KEY],
                    unbiased=unbiased,
                )
                out.append(results)

            else:
                raise NotImplementedError(f"Cannot handle statistics mode {ana_mode}")

        return out

    @staticmethod
    def _per_atom_statistics(
        ana_mode: str,
        arr: torch.Tensor,
        batch: torch.Tensor,
        unbiased: bool = True,
    ):
        """Compute "per-atom" statistics that are normalized by the number of atoms in the system.

        Only makes sense for a graph-level quantity (checked by .statistics).
        """
        # using unique_consecutive handles the non-contiguous selected batch index
        _, N = torch.unique_consecutive(batch, return_counts=True)
        if ana_mode == "mean_std":
            arr = arr / N
            mean = torch.mean(arr)
            std = torch.std(arr, unbiased=unbiased)
            return mean, std
        elif ana_mode == "rms":
            arr = arr / N
            return (torch.sqrt(torch.mean(arr.square())),)
        else:
            raise NotImplementedError(
                f"{ana_mode} for per-atom analysis is not implemented"
            )

    @staticmethod
    def _per_species_statistics(
        ana_mode: str,
        arr: torch.Tensor,
        arr_is_per: str,
        atom_types: torch.Tensor,
        batch: torch.Tensor,
        unbiased: bool = True,
        alpha: Optional[float] = 0.1,
    ):
        """Compute "per-species" statistics.

        For a graph-level quantity, models it as a linear combintation of the number of atoms of different types in the graph.

        For a per-node quantity, computes the expected statistic but for each type instead of over all nodes.
        """
        N = bincount(atom_types, batch)
        N = N[(N > 0).any(dim=1)]  # deal with non-contiguous batch indexes

        if arr_is_per == "graph":

            if ana_mode != "mean_std":
                raise NotImplementedError(
                    f"{ana_mode} for per species analysis is not implemented for shape {arr.shape}"
                )

            N = N.type(torch.get_default_dtype())

            return gp(N, arr, alpha=alpha)

        elif arr_is_per == "node":
            arr = arr.type(torch.get_default_dtype())

            if ana_mode == "mean_std":
                mean = scatter(arr, atom_types, reduce="mean", dim=0)
                std = scatter_std(arr, atom_types, dim=0, unbiased=unbiased)
                return mean, std
            elif ana_mode == "rms":
                square = scatter(arr.square(), atom_types, reduce="mean", dim=0)
                dims = len(square.shape) - 1
                for i in range(dims):
                    square = square.mean(axis=-1)
                return (torch.sqrt(square),)

        else:
            raise NotImplementedError


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
        type_mapper: TypeMapper = None,
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
            type_mapper=type_mapper,
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
            AtomicDataDict.ATOM_TYPE_KEY,
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
        type_mapper: TypeMapper = None,
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
            type_mapper=type_mapper,
        )

    @classmethod
    def from_atoms_list(cls, atoms: Sequence[ase.Atoms], **kwargs):
        """Make an ``ASEDataset`` from a list of ``ase.Atoms`` objects.

        If `root` is not provided, a temporary directory will be used.

        Please note that this is a convinience method that does NOT avoid a round-trip to disk; the provided ``atoms`` will be written out to a file.

        Ignores ``kwargs["file_name"]`` if it is provided.

        Args:
            atoms
            **kwargs: passed through to the constructor
        Returns:
            The constructed ``ASEDataset``.
        """
        if "root" not in kwargs:
            tmpdir = tempfile.TemporaryDirectory()
            kwargs["root"] = tmpdir.name
        else:
            tmpdir = None
        kwargs["file_name"] = tmpdir.name + "/atoms.xyz"
        atoms = list(atoms)
        # Write them out
        ase.io.write(kwargs["file_name"], atoms, format="extxyz")
        # Read them in
        obj = cls(**kwargs)
        if tmpdir is not None:
            # Make it keep a reference to the tmpdir to keep it alive
            # When the dataset is garbage collected, the tmpdir will
            # be too, and will (hopefully) get deleted eventually.
            # Or at least by end of program...
            obj._tmpdir_ref = tmpdir
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
