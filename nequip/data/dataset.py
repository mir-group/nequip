import numpy as np
import logging
import tempfile
import inspect
import functools
import itertools
import yaml
import hashlib
from os.path import dirname, basename, abspath
from typing import Tuple, Dict, Any, List, Callable, Union, Optional, Sequence

import ase
import ase.io

import torch
import torch.multiprocessing as mp

from torch_runstats.scatter import scatter_std, scatter_mean

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
from nequip.utils.regressor import solver
from nequip.utils.savenload import atomic_write
from nequip.utils.multiprocessing import num_tasks
from .transforms import TypeMapper
from .AtomicData import _process_dict


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
        type_mapper (TypeMapper): the transformation to map atomic information to species index. Optional
    """

    def __init__(
        self,
        root: str,
        file_name: Optional[str] = None,
        url: Optional[str] = None,
        force_fixed_keys: List[str] = [],
        extra_fixed_fields: Dict[str, Any] = {},
        include_frames: Optional[List[int]] = None,
        type_mapper: Optional[TypeMapper] = None,
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
            if include_frames is None:
                include_frames = range(num_examples)

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
        _process_dict(fixed_fields, ignore_fields=["r_max"])

        logging.info(f"Loaded data: {data}")

        # use atomic writes to avoid race conditions between
        # different trainings that use the same dataset
        # since those separate trainings should all produce the same results,
        # it doesn't matter if they overwrite each others cached'
        # datasets. It only matters that they don't simultaneously try
        # to write the _same_ file, corrupting it.
        with atomic_write(self.processed_paths[0], binary=True) as f:
            torch.save((data, fixed_fields, self.include_frames), f)
        with atomic_write(self.processed_paths[1], binary=False) as f:
            yaml.dump(self._get_parameters(), f)

        logging.info("Cached processed data to disk")

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
            # note that self._indices is _not_ necessarily in order,
            # while self.data --- which we take our arrays from ---
            # is always in the original order.
            # In particular, the values of `self.data.batch`
            # are indexes in the ORIGINAL order
            # thus we need graph level properties to also be in the original order
            # so that batch values index into them correctly
            # since self.data.batch is always sorted & contiguous
            # (because of Batch.from_data_list)
            # we sort it:
            graph_selector, _ = torch.sort(graph_selector)
        else:
            graph_selector = torch.arange(0, self.len(), stride)
        num_graphs = len(graph_selector)

        node_selector = torch.as_tensor(
            np.in1d(self.data.batch.numpy(), graph_selector.numpy())
        )
        num_nodes = node_selector.sum()

        edge_index = self.data[AtomicDataDict.EDGE_INDEX_KEY]
        edge_selector = node_selector[edge_index[0]] & node_selector[edge_index[1]]
        num_edges = edge_selector.sum()
        del edge_index

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
            elif k == AtomicDataDict.EDGE_INDEX_KEY:
                selectors[k] = (slice(None, None, None), edge_selector)
            elif k in _EDGE_FIELDS:
                selectors[k] = edge_selector
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
            if callable(field):
                # make a joined thing? so it includes fixed fields
                arr, arr_is_per = field(data_transformed)
                arr = arr.to(
                    torch.get_default_dtype()
                )  # all statistics must be on floating
                assert arr_is_per in ("node", "graph", "edge")
            else:
                # Give a better error
                if field not in ff_transformed and field not in data_transformed:
                    raise RuntimeError(
                        f"Field `{field}` for which statistics were requested not found in data."
                    )
                if field not in selectors:
                    # this means field is not selected and so not available
                    raise RuntimeError(
                        f"Only per-node and per-graph fields can have statistics computed; `{field}` has not been registered as either. If it is per-node or per-graph, please register it as such using `nequip.data.register_fields`"
                    )
                if field in ff_transformed:
                    arr = ff_transformed[field]
                else:
                    arr = data_transformed[field]
                if field in _NODE_FIELDS:
                    arr_is_per = "node"
                elif field in _GRAPH_FIELDS:
                    arr_is_per = "graph"
                elif field in _EDGE_FIELDS:
                    arr_is_per = "edge"
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
            elif arr_is_per == "edge":
                arr = arr.view(num_edges, -1)

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
                    algorithm_kwargs=algorithm_kwargs,
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
        N = N.unsqueeze(-1)
        assert N.ndim == 2
        assert N.shape == (len(arr), 1)
        assert arr.ndim >= 2
        data_dim = arr.shape[1:]
        arr = arr / N
        assert arr.shape == (len(N),) + data_dim
        if ana_mode == "mean_std":
            mean = torch.mean(arr, dim=0)
            std = torch.std(arr, unbiased=unbiased, dim=0)
            return mean, std
        elif ana_mode == "rms":
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
        algorithm_kwargs: Optional[dict] = {},
    ):
        """Compute "per-species" statistics.

        For a graph-level quantity, models it as a linear combintation of the number of atoms of different types in the graph.

        For a per-node quantity, computes the expected statistic but for each type instead of over all nodes.
        """
        N = bincount(atom_types.squeeze(-1), batch)
        assert N.ndim == 2  # [batch, n_type]
        N = N[(N > 0).any(dim=1)]  # deal with non-contiguous batch indexes
        assert arr.ndim >= 2
        if arr_is_per == "graph":

            if ana_mode != "mean_std":
                raise NotImplementedError(
                    f"{ana_mode} for per species analysis is not implemented for shape {arr.shape}"
                )

            N = N.type(torch.get_default_dtype())

            return solver(N, arr, **algorithm_kwargs)

        elif arr_is_per == "node":
            arr = arr.type(torch.get_default_dtype())

            if ana_mode == "mean_std":
                mean = scatter_mean(arr, atom_types, dim=0)
                assert mean.shape[1:] == arr.shape[1:]  # [N, dims] -> [type, dims]
                assert len(mean) == N.shape[1]
                std = scatter_std(arr, atom_types, dim=0, unbiased=unbiased)
                assert std.shape == mean.shape
                return mean, std
            elif ana_mode == "rms":
                square = scatter_mean(arr.square(), atom_types, dim=0)
                assert square.shape[1:] == arr.shape[1:]  # [N, dims] -> [type, dims]
                assert len(square) == N.shape[1]
                dims = len(square.shape) - 1
                for i in range(dims):
                    square = square.mean(axis=-1)
                return (torch.sqrt(square),)

        else:
            raise NotImplementedError


# TODO: document fixed field mapped key behavior more clearly
class NpzDataset(AtomicInMemoryDataset):
    """Load data from an npz file.

    To avoid loading unneeded data, keys are ignored by default unless they are in ``key_mapping``, ``include_keys``,
    ``npz_fixed_fields`` or ``extra_fixed_fields``.

    Args:
        key_mapping (Dict[str, str]): mapping of npz keys to ``AtomicData`` keys. Optional
        include_keys (list): the attributes to be processed and stored. Optional
        npz_fixed_field_keys: the attributes that only have one instance but apply to all frames. Optional

    Example: Given a npz file with 10 configurations, each with 14 atoms.

        position: (10, 14, 3)
        force: (10, 14, 3)
        energy: (10,)
        Z: (14)
        user_label1: (10)        # per config
        user_label2: (10, 14, 3) # per atom

    The input yaml should be

    ```yaml
    dataset: npz
    dataset_file_name: example.npz
    include_keys:
      - user_label1
      - user_label2
    npz_fixed_field_keys:
      - cell
      - atomic_numbers
    key_mapping:
      position: pos
      force: forces
      energy: total_energy
      Z: atomic_numbers
    ```

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
        include_keys: List[str] = [],
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
        self.include_keys = include_keys

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

        # only the keys explicitly mentioned in the yaml file will be parsed
        keys = set(list(self.key_mapping.keys()))
        keys.update(self.npz_fixed_field_keys)
        keys.update(self.include_keys)
        keys.update(list(self.extra_fixed_fields.keys()))
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


def _ase_dataset_reader(
    rank: int,
    world_size: int,
    tmpdir: str,
    ase_kwargs: dict,
    atomicdata_kwargs: dict,
    include_frames,
) -> Union[str, List[AtomicData]]:
    """Parallel reader for all frames in file."""
    # interleave--- in theory it is better for performance for the ranks
    # to read consecutive blocks, but the way ASE is written the whole
    # file gets streamed through all ranks anyway, so just trust the OS
    # to cache things sanely, which it will.
    # ASE handles correctly the case where there are no frames in index
    # and just gives an empty list, so that will succeed:
    index = slice(rank, None, world_size)
    if include_frames is None:
        # count includes 0, 1, ..., inf
        include_frames = itertools.count()
    datas = [
        (
            rank + (world_size * i),  # global index
            AtomicData.from_ase(atoms=atoms, **atomicdata_kwargs),
        )
        # include_frames is global indexes, so turn i into a global index
        if rank + (world_size * i) in include_frames
        # in-memory dataset will ignore this later, but needed for indexing to work out
        else None
        # stream them from ase too
        for i, atoms in enumerate(
            ase.io.iread(**ase_kwargs, index=index, parallel=False)
        )
    ]
    # Save to a tempfile---
    # there can be a _lot_ of tensors here, and rather than dealing with
    # the complications of running out of file descriptors and setting
    # sharing methods, since this is a one time thing, just make it simple
    # and avoid shared memory entirely.
    if world_size > 1:
        path = f"{tmpdir}/rank{rank}.pth"
        torch.save(datas, path)
        return path
    else:
        return datas


class ASEDataset(AtomicInMemoryDataset):
    """

    Args:
        ase_args (dict): arguments for ase.io.read
        include_keys (list): in addition to forces and energy, the keys that needs to
             be parsed into dataset
             The data stored in ase.atoms.Atoms.array has the lowest priority,
             and it will be overrided by data in ase.atoms.Atoms.info
             and ase.atoms.Atoms.calc.results. Optional
        key_mapping (dict): rename some of the keys to the value str. Optional

    Example: Given an atomic data stored in "H2.extxyz" that looks like below:

    ```H2.extxyz
    2
    Properties=species:S:1:pos:R:3 energy=-10 user_label=2.0 pbc="F F F"
     H       0.00000000       0.00000000       0.00000000
     H       0.00000000       0.00000000       1.02000000
    ```

    The yaml input should be

    ```
    dataset: ase
    dataset_file_name: H2.extxyz
    ase_args:
      format: extxyz
    include_keys:
      - user_label
    key_mapping:
      user_label: label0
    chemical_symbols:
      - H
    ```

    for VASP parser, the yaml input should be
    ```
    dataset: ase
    dataset_file_name: OUTCAR
    ase_args:
      format: vasp-out
    key_mapping:
      free_energy: total_energy
    chemical_symbols:
      - H
    ```

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
        key_mapping: Optional[dict] = None,
        include_keys: Optional[List[str]] = None,
    ):
        self.ase_args = {}
        self.ase_args.update(getattr(type(self), "ASE_ARGS", dict()))
        self.ase_args.update(ase_args)
        assert "index" not in self.ase_args
        assert "filename" not in self.ase_args

        self.include_keys = include_keys
        self.key_mapping = key_mapping

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

    def get_data(self):
        ase_args = {"filename": self.raw_dir + "/" + self.raw_file_names[0]}
        ase_args.update(self.ase_args)

        # skip the None arguments
        kwargs = dict(
            include_keys=self.include_keys,
            key_mapping=self.key_mapping,
        )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.update(self.extra_fixed_fields)
        n_proc = num_tasks()
        with tempfile.TemporaryDirectory() as tmpdir:
            reader = functools.partial(
                _ase_dataset_reader,
                world_size=n_proc,
                tmpdir=tmpdir,
                ase_kwargs=ase_args,
                atomicdata_kwargs=kwargs,
                include_frames=self.include_frames,
            )
            if n_proc > 1:
                # things hang for some obscure OpenMP reason on some systems when using `fork` method
                ctx = mp.get_context("forkserver")
                with ctx.Pool(processes=n_proc) as p:
                    # map it over the `rank` argument
                    datas = p.map(reader, list(range(n_proc)))
                # clean up the pool before loading the data
                datas = [torch.load(d) for d in datas]
                datas = sum(datas, [])
                # un-interleave the datas
                datas = sorted(datas, key=lambda e: e[0])
            else:
                datas = reader(rank=0)
                # datas here is already in order, stride 1 start 0
                # no need to un-interleave
        # return list of AtomicData:
        return ([e[1] for e in datas],)
