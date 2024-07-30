"""AtomicDataDict

A "static class" (set of functions) operating on `Dict[str, torch.Tensor]`,
aliased here as `AtomicDataDict.Type`.  By avoiding a custom object wrapping
this simple type we avoid unnecessary abstraction and ensure compatibility
with a broad range of PyTorch features and libraries (including TorchScript)
that natively handle dictionaries of tensors.

Each function in this module is sort of like a method on the `AtomicDataDict`
class, if there was such a class---"self" is just passed explicitly:

    AtomicDataDict.some_method(my_data)

Some standard fields:

    pos (Tensor [n_nodes, 3]): Positions of the nodes.
    edge_index (LongTensor [2, n_edges]): ``edge_index[0]`` is the per-edge
        index of the source node and ``edge_index[1]`` is the target node.
    edge_cell_shift (Tensor [n_edges, 3], optional): which periodic image
        of the target point each edge goes to, relative to the source point.
    cell (Tensor [1, 3, 3], optional): the periodic cell for
        ``edge_cell_shift`` as the three triclinic cell vectors.
    node_features (Tensor [n_atom, ...]): the input features of the nodes, optional
    node_attrs (Tensor [n_atom, ...]): the attributes of the nodes, for instance the atom type, optional
    batch (Tensor [n_atom]): the graph to which the node belongs, optional
    atomic_numbers (Tensor [n_atom]): optional
    atom_type (Tensor [n_atom]): optional
"""

from typing import Dict, Union, Tuple, List, Optional
from copy import deepcopy
import warnings

import torch
import ase
from ase.calculators.singlepoint import SinglePointCalculator, SinglePointDFTCalculator
from ase.calculators.calculator import all_properties as ase_all_properties
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress

# Make the keys available in this module
from ._keys import *  # noqa: F403, F401

# Also import the module to use in TorchScript, this is a hack to avoid bug:
# https://github.com/pytorch/pytorch/issues/52312
from . import _keys

from . import _nl, _key_registry

# Define a type alias
Type = Dict[str, torch.Tensor]
# A type representing ASE-style periodic boundary condtions, which can be partial (the tuple case)
PBCType = Union[bool, Tuple[bool, bool, bool]]

# == JIT-unsafe "methods" for general data processing ==


def from_dict(data: dict) -> Type:
    return _key_registry._process_dict(data)


def to_(
    data: Type,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    exclude_keys: List[str] = [],
) -> Type:
    for k, v in data.items():
        if k not in exclude_keys:
            v.to(device=device, dtype=dtype)
    return data


def batched_from_list(data_list: List[Type]) -> Type:
    """Batch multiple AtomicDataDict.Type into one.

    Entries in the input data_list can be batched AtomicDataDict's.
    """
    # == Safety Checks ==
    num_data = len(data_list)
    if num_data == 0:
        raise RuntimeError("Cannot batch empty list of AtomicDataDict.Type")

    # first make sure every AtomicDataDict is batched (even if trivially so)
    # with_batch_() is a no-op if data already has BATCH_KEY and BATCH_PTR_KEY
    data_list = [with_batch_(data) for data in data_list]

    # now every data entry should have BATCH_KEY and BATCH_PTR_KEY
    # check for inconsistent keys over the AtomicDataDicts in the list
    dict_keys = data_list[0].keys()
    assert all(
        [dict_keys == data_list[i].keys() for i in range(len(data_list))]
    ), "Found inconsistent keys across AtomicDataDict list to be batched."
    has_edges = _keys.EDGE_INDEX_KEY in dict_keys

    # == Batching Procedure ==
    out = {}

    # first handle edge indices and batch properties separately
    cum_nodes: int = 0  # for edge indices
    cum_frames: int = 0  # for batch
    edge_idxs = []
    batches = []
    batch_ptrs = []
    for idx in range(num_data):
        if has_edges:
            edge_idxs.append(data_list[idx][_keys.EDGE_INDEX_KEY] + cum_nodes)
        batches.append(data_list[idx][_keys.BATCH_KEY] + cum_frames)
        if idx == 0:
            batch_ptrs.append(data_list[idx][_keys.BATCH_PTR_KEY])
        else:
            batch_ptrs.append(data_list[idx][_keys.BATCH_PTR_KEY][1:] + cum_nodes)
        cum_frames += num_frames(data_list[idx])  # num_frames
        cum_nodes += num_nodes(data_list[idx])  # num_nodes
    if has_edges:
        out[_keys.EDGE_INDEX_KEY] = torch.cat(edge_idxs, dim=1)  # (2, num_edges)
    out[_keys.BATCH_KEY] = torch.cat(batches, dim=0)
    out[_keys.BATCH_PTR_KEY] = torch.cat(batch_ptrs, dim=0)

    # then collect the rest by field...
    batched = {}
    for k in dict_keys:
        # ignore these since handled previously
        if k in [_keys.EDGE_INDEX_KEY, _keys.BATCH_KEY, _keys.BATCH_PTR_KEY]:
            continue
        batched[k] = [d[k] for d in data_list]
    # ... and concatenate
    for k, vs in batched.items():
        if k in _key_registry._GRAPH_FIELDS:
            # graph-level properties and so need a new batch dimension
            out[k] = torch.cat(vs, dim=0)
        elif k in (_key_registry._NODE_FIELDS | _key_registry._EDGE_FIELDS):
            # cat along the node or edge dim (dim 0)
            out[k] = torch.cat(vs, dim=0)
        else:
            # TODO: rethink this
            if k not in ase_all_properties:
                raise KeyError(f"Unregistered key {k}")

    return out


def frame_from_batched(batched_data: Type, index: int) -> Type:
    """Returns a single frame from batched data."""
    # get data with batches just in case this is called on unbatched data
    batched_data = with_batch_(batched_data)
    # use zero-indexing as per python norm
    N_frames = num_frames(batched_data)
    assert index < num_frames(
        batched_data
    ), f"Input data consists of {N_frames} frames so index can run from 0 to {N_frames-1} -- but given index of {index}!"
    batches = batched_data[_keys.BATCH_KEY]
    node_idx_offset = batched_data[_keys.BATCH_PTR_KEY][index]
    if _keys.EDGE_INDEX_KEY in batched_data:
        edge_center_idx = batched_data[_keys.EDGE_INDEX_KEY][0]

    out = {}
    for k, v in batched_data.items():
        #  will handle batch properties later
        if k in [_keys.BATCH_KEY, _keys.BATCH_PTR_KEY]:
            continue
        elif k == _keys.EDGE_INDEX_KEY:
            # special case since shape is (2, num_edges), and to remove edge index offset
            out[k] = (
                torch.masked_select(v, torch.eq(batches[edge_center_idx], index)).view(
                    2, -1
                )
                - node_idx_offset
            )
        elif k in _key_registry._GRAPH_FIELDS:
            # index out relevant frame
            out[k] = v[[index]]  # to ensure batch dimension remains
        elif k in _key_registry._NODE_FIELDS:
            # mask out relevant portion
            out[k] = v[batches == index]
        elif k in _key_registry._EDGE_FIELDS:  # excluding edge indices
            out[k] = v[torch.eq(batches[edge_center_idx], index)]
        else:
            # TODO: rethink this
            if k not in ase_all_properties:
                raise KeyError(f"Unregistered key {k}")

    # account for batch information
    out = with_batch_(out)
    return out


def from_ase(
    atoms: ase.Atoms,
    key_mapping: Optional[Dict[str, str]] = {},
    include_keys: Optional[list] = [],
) -> Type:
    """Build a ``AtomicDataDict`` from an ``ase.Atoms`` object.

    Respects ``atoms``'s ``pbc`` and ``cell``.

    First tries to extract energies and forces from a single-point calculator associated with the ``Atoms`` if one is present and has those fields.
    If either is not found, the method will look for ``energy``/``energies`` and ``force``/``forces`` in ``atoms.arrays``.

    `get_atomic_numbers()` will be stored as the atomic_numbers attribute.

    Args:
        atoms (ase.Atoms): the input.
        features (torch.Tensor shape [N, M], optional): per-atom M-dimensional feature vectors. If ``None`` (the
            default), uses a one-hot encoding of the species present in ``atoms``.
        include_keys (list): list of additional keys to include in AtomicData aside from the ones defined in
                ase.calculators.calculator.all_properties. Optional
        key_mapping (dict): rename ase property name to a new string name. Optional

    Returns:
        A ``AtomicData``.
    """
    from nequip.ase import NequIPCalculator

    default_args = set(
        [
            "numbers",
            "positions",
        ]  # ase internal names for position and atomic_numbers
        + [
            "pbc",
            _keys.CELL_KEY,
            _keys.POSITIONS_KEY,
        ]  # arguments for from_dict method
    )
    include_keys = list(
        set(include_keys + ase_all_properties + list(key_mapping.keys())) - default_args
    )

    km = {
        "forces": _keys.FORCE_KEY,
        "energy": _keys.TOTAL_ENERGY_KEY,
    }
    km.update(key_mapping)
    key_mapping = km

    add_fields = {}

    # Get info from atoms.arrays; lowest priority. copy first
    add_fields = {
        key_mapping.get(k, k): v for k, v in atoms.arrays.items() if k in include_keys
    }

    # Get info from atoms.info; second lowest priority.
    add_fields.update(
        {key_mapping.get(k, k): v for k, v in atoms.info.items() if k in include_keys}
    )

    if atoms.calc is not None:
        if isinstance(atoms.calc, (SinglePointCalculator, SinglePointDFTCalculator)):
            add_fields.update(
                {
                    key_mapping.get(k, k): deepcopy(v)
                    for k, v in atoms.calc.results.items()
                    if k in include_keys
                }
            )
        elif isinstance(atoms.calc, NequIPCalculator):
            pass  # otherwise the calculator breaks
        else:
            raise NotImplementedError(
                f"`from_ase` does not support calculator {atoms.calc}"
            )

    add_fields[_keys.ATOMIC_NUMBERS_KEY] = atoms.get_atomic_numbers()

    cell = atoms.get_cell()
    pbc = atoms.pbc
    # IMPORTANT: the following reshape logic only applies to rank-2 Cartesian tensor fields
    for key in add_fields:
        if key in _key_registry._CARTESIAN_TENSOR_FIELDS:
            # enforce (3, 3) shape for graph fields, e.g. stress, virial
            if key in _key_registry._GRAPH_FIELDS:
                # handle ASE-style 6 element Voigt order stress
                if key in (_keys.STRESS_KEY, _keys.VIRIAL_KEY):
                    if add_fields[key].shape == (6,):
                        add_fields[key] = voigt_6_to_full_3x3_stress(add_fields[key])
                if add_fields[key].shape == (3, 3):
                    # it's already 3x3, do nothing else
                    pass
                elif add_fields[key].shape == (9,):
                    add_fields[key] = add_fields[key].reshape((3, 3))
                else:
                    raise RuntimeError(
                        f"bad shape for {key} registered as a Cartesian tensor graph field---please note that only rank-2 Cartesian tensors are currently supported"
                    )
            # enforce (N_atom, 3, 3) shape for node fields, e.g. Born effective charges
            elif key in _key_registry._NODE_FIELDS:
                if add_fields[key].shape[1:] == (3, 3):
                    pass
                elif add_fields[key].shape[1:] == (9,):
                    add_fields[key] = add_fields[key].reshape((-1, 3, 3))
                else:
                    raise RuntimeError(
                        f"bad shape for {key} registered as a Cartesian tensor node field---please note that only rank-2 Cartesian tensors are currently supported"
                    )
            else:
                raise RuntimeError(
                    f"{key} registered as a Cartesian tensor field was not registered as either a graph or node field"
                )
    data = {
        _keys.POSITIONS_KEY: atoms.positions,
        _keys.CELL_KEY: cell,
        _keys.PBC_KEY: pbc,
    }
    data.update(**add_fields)

    return with_batch_(from_dict(data))


# TODO: this can potentially be cleaner if we iterate and use frame_from_batched()
def to_ase(
    data: Type,
    type_mapper=None,
    extra_fields: List[str] = [],
) -> Union[List[ase.Atoms], ase.Atoms]:
    """Build a (list of) ``ase.Atoms`` object(s) from an ``AtomicData`` object.

    For each unique batch number provided in ``AtomicDataDict.BATCH_KEY``,
    an ``ase.Atoms`` object is created. If ``AtomicDataDict.BATCH_KEY`` does not
    exist in self, a single ``ase.Atoms`` object is created.

    Args:
        type_mapper: if provided, will be used to map ``ATOM_TYPES`` back into
            elements, if the configuration of the ``type_mapper`` allows.
        extra_fields: fields other than those handled explicitly (currently
            those defining the structure as well as energy, per-atom energy,
            and forces) to include in the output object. Per-atom (per-node)
            quantities will be included in ``arrays``; per-graph and per-edge
            quantities will be included in ``info``.

    Returns:
        A list of ``ase.Atoms`` objects if ``AtomicDataDict.BATCH_KEY`` is in self
        and is not None. Otherwise, a single ``ase.Atoms`` object is returned.
    """
    positions = data[_keys.POSITIONS_KEY]
    edge_index = data[_keys.EDGE_INDEX_KEY]
    if positions.device != torch.device("cpu"):
        raise TypeError(
            "Explicitly move this `AtomicData` to CPU using `.to()` before calling `to_ase()`."
        )
    if _keys.ATOMIC_NUMBERS_KEY in data:
        atomic_nums = data[_keys.ATOMIC_NUMBERS_KEY]
    elif type_mapper is not None and type_mapper.has_chemical_symbols:
        atomic_nums = type_mapper.untransform(data[_keys.ATOM_TYPE_KEY])
    else:
        warnings.warn(
            "AtomicData.to_ase(): data didn't contain atomic numbers... using atom_type as atomic numbers instead, but this means the chemical symbols in ASE (outputs) will be wrong"
        )
        atomic_nums = data[_keys.ATOM_TYPE_KEY]
    pbc = data.get(_keys.PBC_KEY, None)
    cell = data.get(_keys.CELL_KEY, None)
    batch = data.get(_keys.BATCH_KEY, None)
    energy = data.get(_keys.TOTAL_ENERGY_KEY, None)
    energies = data.get(_keys.PER_ATOM_ENERGY_KEY, None)
    force = data.get(_keys.FORCE_KEY, None)
    do_calc = any(
        k in data
        for k in [
            _keys.TOTAL_ENERGY_KEY,
            _keys.FORCE_KEY,
            _keys.PER_ATOM_ENERGY_KEY,
            _keys.STRESS_KEY,
        ]
    )

    # exclude those that are special for ASE and that we process seperately
    special_handling_keys = [
        _keys.POSITIONS_KEY,
        _keys.CELL_KEY,
        _keys.PBC_KEY,
        _keys.ATOMIC_NUMBERS_KEY,
        _keys.TOTAL_ENERGY_KEY,
        _keys.FORCE_KEY,
        _keys.PER_ATOM_ENERGY_KEY,
        _keys.STRESS_KEY,
    ]
    assert (
        len(set(extra_fields).intersection(special_handling_keys)) == 0
    ), f"Cannot specify keys handled in special ways ({special_handling_keys}) as `extra_fields` for atoms output--- they are output by default"

    if cell is not None:
        cell = cell.view(-1, 3, 3)
    if pbc is not None:
        pbc = pbc.view(-1, 3)

    if batch is not None:
        n_batches = batch.max() + 1
        cell = cell.expand(n_batches, 3, 3) if cell is not None else None
        pbc = pbc.expand(n_batches, 3) if pbc is not None else None
    else:
        n_batches = 1

    batch_atoms = []
    for batch_idx in range(n_batches):
        if batch is not None:
            mask = batch == batch_idx
            mask = mask.view(-1)
            # if both ends of the edge are in the batch, the edge is in the batch
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        else:
            mask = slice(None)
            edge_mask = slice(None)

        mol = ase.Atoms(
            numbers=atomic_nums[mask].view(-1),  # must be flat for ASE
            positions=positions[mask],
            cell=cell[batch_idx] if cell is not None else None,
            pbc=pbc[batch_idx] if pbc is not None else None,
        )

        if do_calc:
            fields = {}
            if energies is not None:
                fields["energies"] = energies[mask].cpu().numpy()
            if energy is not None:
                fields["energy"] = energy[batch_idx].cpu().numpy()
            if force is not None:
                fields["forces"] = force[mask].cpu().numpy()
            if _keys.STRESS_KEY in data:
                fields["stress"] = full_3x3_to_voigt_6_stress(
                    data["stress"].view(-1, 3, 3)[batch_idx].cpu().numpy()
                )
            mol.calc = SinglePointCalculator(mol, **fields)

        # add other information
        for key in extra_fields:
            if key in _key_registry._NODE_FIELDS:
                # mask it
                mol.arrays[key] = data[key][mask].cpu().numpy().reshape(mask.sum(), -1)
            elif key in _key_registry._EDGE_FIELDS:
                mol.info[key] = (
                    data[key][edge_mask].cpu().numpy().reshape(edge_mask.sum(), -1)
                )
            elif key == _keys.EDGE_INDEX_KEY:
                mol.info[key] = data[key][:, edge_mask].cpu().numpy()
            elif key in _key_registry._GRAPH_FIELDS:
                mol.info[key] = data[key][batch_idx].cpu().numpy().reshape(-1)
            else:
                raise RuntimeError(
                    f"Extra field `{key}` isn't registered as node/edge/graph"
                )

        batch_atoms.append(mol)

    if batch is not None:
        return batch_atoms
    else:
        assert len(batch_atoms) == 1
        return batch_atoms[0]


def compute_neighborlist_(data: Type, r_max: float, **kwargs) -> Type:
    """Add a neighborlist to `data` in-place.

    This can be called on alredy-batched data.
    """
    to_batch = []
    data = with_batch_(data)
    for idx in range(num_frames(data)):
        data_per_frame = frame_from_batched(data, idx)

        cell = data_per_frame.get(_keys.CELL_KEY, None)
        if cell is not None:
            cell = cell.view(3, 3)  # remove batch dimension

        pbc = data_per_frame.get(_keys.PBC_KEY, None)
        if pbc is not None:
            pbc = pbc.view(3)  # remove batch dimension

        edge_index, edge_cell_shift, cell = _nl.neighbor_list_and_relative_vec(
            pos=data_per_frame[_keys.POSITIONS_KEY],
            r_max=r_max,
            cell=cell,
            pbc=pbc,
            **kwargs,
        )
        # add neighborlist information
        data_per_frame[_keys.EDGE_INDEX_KEY] = edge_index
        if data.get(_keys.CELL_KEY, None) is not None and edge_cell_shift is not None:
            data_per_frame[_keys.EDGE_CELL_SHIFT_KEY] = edge_cell_shift
        to_batch.append(data_per_frame)

    # rebatch to make sure neighborlist information is in a similar batched format
    return batched_from_list(to_batch)


def without_nodes(data: Type, which_nodes: torch.Tensor) -> Type:
    """Return a copy of ``data`` with ``which_nodes`` removed.
    The returned object may share references to some underlying data tensors with ``data``.

    Args:
        data (AtomicDataDict)     : atomic data dict
        which_nodes (torch.Tensor): index tensor or boolean mask

    Returns:
        A new data object.
    """
    N_nodes = num_nodes(data)
    which_nodes = torch.as_tensor(which_nodes)
    if which_nodes.dtype == torch.bool:
        node_mask = ~which_nodes
    else:
        node_mask = torch.ones(N_nodes, dtype=torch.bool)
        node_mask[which_nodes] = False
    assert node_mask.shape == (N_nodes,)
    n_keeping = node_mask.sum()

    # Only keep edges where both from and to are kept
    edge_idx = data[_keys.EDGE_INDEX_KEY]
    edge_mask = node_mask[edge_idx[0]] & node_mask[edge_idx[1]]
    # Create an index mapping:
    new_index = torch.full((N_nodes,), -1, dtype=torch.long)
    new_index[node_mask] = torch.arange(n_keeping, dtype=torch.long)

    new_dict = {}
    for k, v in data.items():
        if k == _keys.EDGE_INDEX_KEY:
            new_dict[k] = new_index[v[:, edge_mask]]
        elif k in _key_registry._GRAPH_FIELDS:
            new_dict[k] = v
        elif k in _key_registry._NODE_FIELDS:
            new_dict[k] = v[node_mask]
        elif k in _key_registry._EDGE_FIELDS:
            new_dict[k] = v[edge_mask]
        else:
            raise KeyError(f"Unregistered key {k}")

    return new_dict


# == JIT-safe "methods" for use in model code ==


@torch.jit.script
def num_frames(data: Type) -> int:
    # will not check if batch information is present
    return len(data[_keys.BATCH_PTR_KEY]) - 1


@torch.jit.script
def num_nodes(data: Type) -> int:
    return len(data[_keys.POSITIONS_KEY])


@torch.jit.script
def num_edges(data: Type) -> int:
    # will not check if neighborlist is present
    return data[_keys.EDGE_INDEX_KEY].shape[1]


@torch.jit.script
def with_edge_vectors(
    data: Type,
    with_lengths: bool = True,
    edge_index_field: str = _keys.EDGE_INDEX_KEY,
    edge_cell_shift_field: str = _keys.EDGE_CELL_SHIFT_KEY,
    edge_vec_field: str = _keys.EDGE_VECTORS_KEY,
    edge_vec_f64_field: str = _keys.EDGE_VECTORS_F64_KEY,
    edge_len_field: str = _keys.EDGE_LENGTH_KEY,
    edge_len_f64_field: str = _keys.EDGE_LENGTH_F64_KEY,
) -> Type:
    """Compute the edge displacement vectors for a graph.

    If ``data.pos.requires_grad`` and/or ``data.cell.requires_grad``, this
    method will return edge vectors correctly connected in the autograd graph.

    The availability of various custom field options enables reuse of this function
    for nonconventional field options.

    Returns:
        Tensor [n_edges, 3] edge displacement vectors
    """
    # if present in AtomicDataDict, use MODEL_DTYPE_KEY; otherwise, use dtype of positions
    model_dtype: torch.dtype = data.get(
        _keys.MODEL_DTYPE_KEY, data[_keys.POSITIONS_KEY]
    ).dtype
    # We do calculations on the positions and cells in whatever dtype they
    # were provided in, and only convert to model_dtype after
    if edge_vec_field in data:
        if with_lengths and edge_len_field not in data:
            edge_len = torch.linalg.norm(data[edge_vec_f64_field], dim=-1)
            data[edge_len_f64_field] = edge_len
            data[edge_len_field] = edge_len.to(model_dtype)
        return data
    else:
        # Build it dynamically
        # Note that this is
        # (1) backwardable, because everything (pos, cell, shifts)
        #     is Tensors.
        # (2) works on a Batch constructed from AtomicData
        pos = data[_keys.POSITIONS_KEY]
        edge_index = data[edge_index_field]
        edge_vec = torch.index_select(pos, 0, edge_index[1]) - torch.index_select(
            pos, 0, edge_index[0]
        )
        if _keys.CELL_KEY in data:
            # ^ note that to save time we don't check that the edge_cell_shifts are trivial if no cell is provided; we just assume they are either not present or all zero.
            # -1 gives a batch dim no matter what
            cell = data[_keys.CELL_KEY].view(-1, 3, 3)
            edge_cell_shift = data[edge_cell_shift_field]
            if cell.shape[0] > 1:
                batch = data[_keys.BATCH_KEY]
                # Cell has a batch dimension
                # note the ASE cell vectors as rows convention
                edge_vec = edge_vec + torch.einsum(
                    "ni,nij->nj",
                    edge_cell_shift,
                    cell[batch[edge_index[0]]],
                )
                # TODO: is there a more efficient way to do the above without
                # creating an [n_edge] and [n_edge, 3, 3] tensor?
            else:
                # Cell has either no batch dimension, or a useless one,
                # so we can avoid creating the large intermediate cell tensor.
                # Note that we do NOT check that the batch array, if it is present,
                # is trivial — but this does need to be consistent.
                edge_vec = edge_vec + torch.einsum(
                    "ni,ij->nj",
                    edge_cell_shift,
                    cell.squeeze(0),  # remove batch dimension
                )
        data[edge_vec_f64_field] = edge_vec
        data[edge_vec_field] = edge_vec.to(model_dtype)
        if with_lengths:
            edge_len = torch.linalg.norm(edge_vec, dim=-1)
            data[edge_len_f64_field] = edge_len
            data[edge_len_field] = edge_len.to(model_dtype)
        return data


@torch.jit.script
def with_batch_(data: Type) -> Type:
    """Get batch Tensor.

    If this AtomicDataPrimitive has no ``batch``, one of all zeros will be
    allocated and returned.
    """
    if _keys.BATCH_KEY in data:
        assert _keys.BATCH_PTR_KEY in data
        return data
    else:
        pos = data[_keys.POSITIONS_KEY]
        batch = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
        data[_keys.BATCH_KEY] = batch
        # ugly way to make a tensor of [0, len(pos)], but it avoids transfers or casts
        data[_keys.BATCH_PTR_KEY] = torch.arange(
            start=0,
            end=len(pos) + 1,
            step=len(pos),
            dtype=torch.long,
            device=pos.device,
        )
        return data


# For autocomplete in IDEs, don't expose our various imports
__all__ = [
    to_,
    from_dict,
    from_ase,
    without_nodes,
    compute_neighborlist_,
    num_nodes,
    num_edges,
    with_batch_,
    with_edge_vectors,
] + _keys.ALLOWED_KEYS
