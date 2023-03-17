import numpy as np
import logging
import inspect
import itertools
import yaml
import hashlib
import math
from typing import Tuple, Dict, Any, List, Callable, Union, Optional

import torch

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
from ..transforms import TypeMapper


class AtomicDataset(Dataset):
    """The base class for all NequIP datasets."""

    root: str
    dtype: torch.dtype

    def __init__(
        self,
        root: str,
        type_mapper: Optional[TypeMapper] = None,
    ):
        self.dtype = torch.get_default_dtype()
        super().__init__(root=root, transform=type_mapper)

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
        params["dtype"] = str(self.dtype)
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
        AtomicData_options (dict, optional): extra key that are not stored in data but needed for AtomicData initialization
        include_frames (list, optional): the frames to process with the constructor.
        type_mapper (TypeMapper): the transformation to map atomic information to species index. Optional
    """

    def __init__(
        self,
        root: str,
        file_name: Optional[str] = None,
        url: Optional[str] = None,
        AtomicData_options: Dict[str, Any] = {},
        include_frames: Optional[List[int]] = None,
        type_mapper: Optional[TypeMapper] = None,
    ):
        # TO DO, this may be simplified
        # See if a subclass defines some inputs
        self.file_name = (
            getattr(type(self), "FILE_NAME", None) if file_name is None else file_name
        )
        self.url = getattr(type(self), "URL", url)

        self.AtomicData_options = AtomicData_options
        self.include_frames = include_frames

        self.data = None

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
        super().__init__(root=root, type_mapper=type_mapper)
        if self.data is None:
            self.data, include_frames = torch.load(self.processed_paths[0])
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

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pth", "params.yaml"]

    def get_data(
        self,
    ) -> Union[Tuple[Dict[str, Any], Dict[str, Any]], List[AtomicData]]:
        """Get the data --- called from ``process()``, can assume that ``raw_file_names()`` exist.

        Note that parameters for graph construction such as ``pbc`` and ``r_max`` should be included here as (likely, but not necessarily, fixed) fields.

        Returns:
        A dict:
            fields: dict
                mapping a field name ('pos', 'cell') to a list-like sequence of tensor-like objects giving that field's value for each example.
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
        if isinstance(data, list):

            # It's a data list
            data_list = data
            if not (self.include_frames is None or data_list is None):
                data_list = [data_list[i] for i in self.include_frames]
            assert all(isinstance(e, AtomicData) for e in data_list)
            assert all(AtomicDataDict.BATCH_KEY not in e for e in data_list)

            fields = {}

        elif isinstance(data, dict):
            # It's fields
            # Get our data
            fields = data

            # check keys
            all_keys = set(fields.keys())
            assert AtomicDataDict.BATCH_KEY not in all_keys
            # Check bad key combinations, but don't require that this be a graph yet.
            AtomicDataDict.validate_keys(all_keys, graph_required=False)

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
                assert "r_max" in self.AtomicData_options
                assert AtomicDataDict.POSITIONS_KEY in all_keys

            data_list = [
                constructor(
                    **{
                        **{f: v[i] for f, v in fields.items()},
                        **self.AtomicData_options,
                    }
                )
                for i in include_frames
            ]

        else:
            raise ValueError("Invalid return from `self.get_data()`")

        # Batch it for efficient saving
        # This limits an AtomicInMemoryDataset to a maximum of LONG_MAX atoms _overall_, but that is a very big number and any dataset that large is probably not "InMemory" anyway
        data = Batch.from_data_list(data_list)
        del data_list
        del fields

        total_MBs = sum(item.numel() * item.element_size() for _, item in data) / (
            1024 * 1024
        )
        logging.info(
            f"Loaded data: {data}\n    processed data size: ~{total_MBs:.2f} MB"
        )
        del total_MBs

        # use atomic writes to avoid race conditions between
        # different trainings that use the same dataset
        # since those separate trainings should all produce the same results,
        # it doesn't matter if they overwrite each others cached'
        # datasets. It only matters that they don't simultaneously try
        # to write the _same_ file, corrupting it.
        with atomic_write(self.processed_paths[0], binary=True) as f:
            torch.save((data, self.include_frames), f)
        with atomic_write(self.processed_paths[1], binary=False) as f:
            yaml.dump(self._get_parameters(), f)

        logging.info("Cached processed data to disk")

        self.data = data

    def get(self, idx):
        return self.data.get_example(idx)

    def _selectors(
        self,
        stride: int = 1,
    ):
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

        node_selector = torch.as_tensor(
            np.in1d(self.data.batch.numpy(), graph_selector.numpy())
        )

        edge_index = self.data[AtomicDataDict.EDGE_INDEX_KEY]
        edge_selector = node_selector[edge_index[0]] & node_selector[edge_index[1]]

        return (graph_selector, node_selector, edge_selector)

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

            modes: the statistic to compute for each field. Valid options are:
                 - ``count``
                 - ``rms``
                 - ``mean_std``
                 - ``per_atom_*``
                 - ``per_species_*``

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

        graph_selector, node_selector, edge_selector = self._selectors(stride=stride)

        num_graphs = len(graph_selector)
        num_nodes = node_selector.sum()
        num_edges = edge_selector.sum()

        if self.transform is not None:
            # pre-transform the data so that statistics process transformed data
            data_transformed = self.transform(self.data.to_dict(), types_required=False)
        else:
            data_transformed = self.data.to_dict()
        # pre-select arrays
        # this ensures that all following computations use the right data
        all_keys = set()
        selectors = {}
        for k in data_transformed.keys():
            all_keys.add(k)
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
                arr = arr.to(self.dtype)  # all statistics must be on floating
                assert arr_is_per in ("node", "graph", "edge")
            else:
                if field not in all_keys:
                    raise RuntimeError(
                        f"The field key `{field}` is not present in this dataset"
                    )
                if field not in selectors:
                    # this means field is not selected and so not available
                    raise RuntimeError(
                        f"Only per-node and per-graph fields can have statistics computed; `{field}` has not been registered as either. If it is per-node or per-graph, please register it as such using `nequip.data.register_fields`"
                    )
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
                    arr = torch.as_tensor(arr, dtype=self.dtype)
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
                if len(arr) < 2:
                    raise ValueError(
                        "Can't do per species standard deviation without at least two samples"
                    )
                mean = torch.mean(arr, dim=0)
                std = torch.std(arr, dim=0, unbiased=unbiased)
                out.append((mean, std))

            elif ana_mode == "absmax":
                out.append((arr.abs().max(),))

            elif ana_mode.startswith("per_species_"):
                # per-species
                algorithm_kwargs = kwargs.pop(field + ana_mode, {})

                ana_mode = ana_mode[len("per_species_") :]

                if atom_types is None:
                    atom_types = data_transformed[AtomicDataDict.ATOM_TYPE_KEY]

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
            if len(arr) < 2:
                raise ValueError(
                    "Can't do standard deviation without at least two samples"
                )
            mean = torch.mean(arr, dim=0)
            std = torch.std(arr, unbiased=unbiased, dim=0)
            return mean, std
        elif ana_mode == "rms":
            return (torch.sqrt(torch.mean(arr.square())),)
        elif ana_mode == "absmax":
            return (torch.max(arr.abs()),)
        else:
            raise NotImplementedError(
                f"{ana_mode} for per-atom analysis is not implemented"
            )

    def _per_species_statistics(
        self,
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

            N = N.type(self.dtype)

            return solver(N, arr, **algorithm_kwargs)

        elif arr_is_per == "node":
            arr = arr.type(self.dtype)

            if ana_mode == "mean_std":
                # There need to be at least two occurances of each atom type in the
                # WHOLE dataset, not in any given frame:
                if torch.any(N.sum(dim=0) < 2):
                    raise ValueError(
                        "Can't do per species standard deviation without at least two samples per species"
                    )
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
                raise NotImplementedError(
                    f"Statistics mode {ana_mode} isn't yet implemented for per_species_"
                )

        else:
            raise NotImplementedError

    def rdf(
        self, bin_width: float, stride: int = 1
    ) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
        """Compute the pairwise RDFs of the dataset.

        Args:
            bin_width: width of the histogram bin in distance units
            stride: stride of data to include

        Returns:
            dictionary mapping `(type1, type2)` to tuples of `(hist, bin_edges)` in the style of `np.histogram`.
        """
        graph_selector, node_selector, edge_selector = self._selectors(stride=stride)

        data = AtomicData.to_AtomicDataDict(self.data)
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

        results = {}

        types = self.type_mapper(data)[AtomicDataDict.ATOM_TYPE_KEY]

        edge_types = torch.index_select(
            types, 0, data[AtomicDataDict.EDGE_INDEX_KEY].reshape(-1)
        ).view(2, -1)
        types_center = edge_types[0].numpy()
        types_neigh = edge_types[1].numpy()

        r_max: float = self.AtomicData_options["r_max"]
        # + 1 to always have a zero bin at the end
        n_bins: int = int(math.ceil(r_max / bin_width)) + 1
        # +1 since these are bin_edges including rightmost
        bins = bin_width * np.arange(n_bins + 1)

        for type1, type2 in itertools.combinations_with_replacement(
            range(self.type_mapper.num_types), 2
        ):
            # Try to do as much of this as possible in-place
            mask = types_center == type1
            np.logical_and(mask, types_neigh == type2, out=mask)
            np.logical_and(mask, edge_selector, out=mask)
            mask = mask.astype(np.int32)
            results[(type1, type2)] = np.histogram(
                data[AtomicDataDict.EDGE_LENGTH_KEY],
                weights=mask,
                bins=bins,
                density=True,
            )
            # RDF is symmetric
            results[(type2, type1)] = results[(type1, type2)]

        return results
