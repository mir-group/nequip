"""AtomicData: neighbor graphs in (periodic) real space.

Authors: Albert Musaelian
"""

import warnings
from copy import deepcopy
from typing import Union, Tuple, Dict, Optional, List
from collections.abc import Mapping

import numpy as np
import ase.neighborlist
import ase
from ase.calculators.singlepoint import SinglePointCalculator, SinglePointDFTCalculator

import torch
from torch_geometric.data import Data
import e3nn.o3

from . import AtomicDataDict
from ._util import _TORCH_INTEGER_DTYPES

# A type representing ASE-style periodic boundary condtions, which can be partial (the tuple case)
PBC = Union[bool, Tuple[bool, bool, bool]]


class AtomicData(Data):
    """A neighbor graph for points in (periodic triclinic) real space.

    For typical cases either ``from_points`` or ``from_ase`` should be used to
    construct a AtomicData; they also standardize and check their input much more
    thoroughly.

    In general, ``node_features`` are features or input information on the nodes that will be fed through and transformed by the network, while ``node_attrs`` are _encodings_ fixed, inherant attributes of the atoms themselves that remain constant through the network.
    For example, a one-hot _encoding_ of atomic species is a node attribute, while some observed instantaneous property of that atom (current partial charge, for example), would be a feature.

    In general, ``torch.Tensor`` arguments should be of consistant dtype. Numpy arrays will be converted to ``torch.Tensor``s; those of floating point dtype will be converted to ``torch.get_current_dtype()`` regardless of their original precision. Scalar values (Python scalars or ``torch.Tensor``s of shape ``()``) a resized to tensors of shape ``[1]``. Per-atom scalar values should be given with shape ``[N_at, 1]``.

    ``AtomicData`` should be used for all data creation and manipulation outside of the model; inside of the model ``AtomicDataDict.Type`` is used.

    Args:
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
        atomic_numbers (Tensor [n_atom]): optional.
        species_index (Tensor [n_atom]): optional.
        **kwargs: other data, optional.
    """

    def __init__(self, irreps: Dict[str, e3nn.o3.Irreps] = {}, **kwargs):

        # empty init needed by get_example
        if len(kwargs) == 0 and len(irreps) == 0:
            super().__init__()
            return

        # Check the keys
        AtomicDataDict.validate_keys(kwargs)
        # Deal with _some_ dtype issues
        for k, v in kwargs.items():
            if (
                k == AtomicDataDict.EDGE_INDEX_KEY
                or k == AtomicDataDict.ATOMIC_NUMBERS_KEY
                or k == AtomicDataDict.SPECIES_INDEX_KEY
                or k == AtomicDataDict.BATCH_KEY
            ):
                # Any property used as an index must be long (or byte or bool, but those are not relevant for atomic scale systems)
                # int32 would pass later checks, but is actually disallowed by torch
                kwargs[k] = torch.as_tensor(v, dtype=torch.long)
            elif isinstance(v, np.ndarray):
                if np.issubdtype(v.dtype, np.floating):
                    kwargs[k] = torch.as_tensor(v, dtype=torch.get_default_dtype())
                else:
                    kwargs[k] = torch.as_tensor(v)
            elif np.issubdtype(type(v), np.floating):
                # Force scalars to be tensors with a data dimension
                # This makes them play well with irreps
                kwargs[k] = torch.as_tensor(
                    v, dtype=torch.get_default_dtype()
                ).unsqueeze(-1)
            elif isinstance(v, torch.Tensor) and len(v.shape) == 0:
                # ^ this tensor is a scalar; we need to give it
                # a data dimension to play nice with irreps
                kwargs[k] = v.unsqueeze(-1)

        super().__init__(num_nodes=len(kwargs["pos"]), **kwargs)

        # Validate shapes
        assert self.pos.dim() == 2 and self.pos.shape[1] == 3
        assert self.edge_index.dim() == 2 and self.edge_index.shape[0] == 2
        if "edge_cell_shift" in self and self.edge_cell_shift is not None:
            assert self.edge_cell_shift.shape == (self.num_edges, 3)
            assert self.edge_cell_shift.dtype == self.pos.dtype
        if "cell" in self and self.cell is not None:
            assert (self.cell.shape == (3, 3)) or (
                self.cell.dim() == 3 and self.cell.shape[1:] == (3, 3)
            )
            assert self.cell.dtype == self.pos.dtype
        if "node_features" in self and self.node_features is not None:
            assert self.node_features.shape[0] == self.num_nodes
            assert self.node_features.dtype == self.pos.dtype
        if "node_attrs" in self and self.node_attrs is not None:
            assert self.node_attrs.shape[0] == self.num_nodes
            assert self.node_attrs.dtype == self.pos.dtype

        if (
            AtomicDataDict.ATOMIC_NUMBERS_KEY in self
            and self.atomic_numbers is not None
        ):
            assert self.atomic_numbers.dtype in _TORCH_INTEGER_DTYPES
        if "batch" in self and self.batch is not None:
            assert self.batch.dim() == 1 and self.batch.shape[0] == self.num_nodes
            # Check that there are the right number of cells
            if "cell" in self and self.cell is not None:
                cell = self.cell.view(-1, 3, 3)
                assert cell.shape[0] == self.batch.max() + 1

        # Validate irreps
        # __*__ is the only way to hide from torch_geometric
        self.__irreps__ = AtomicDataDict._fix_irreps_dict(irreps)
        for field, irreps in self.__irreps__:
            if irreps is not None:
                assert self[field].shape[-1] == irreps.dim

    @classmethod
    def from_points(
        cls,
        pos=None,
        r_max: float = None,
        self_interaction: bool = False,
        strict_self_interaction: bool = True,
        cell=None,
        pbc: Optional[PBC] = None,
        **kwargs,
    ):
        """Build neighbor graph from points, optionally with PBC.

        Args:
            pos (np.ndarray/torch.Tensor shape [N, 3]): node positions. If Tensor, must be on the CPU.
            r_max (float): neighbor cutoff radius.
            cell (ase.Cell/ndarray [3,3], optional): periodic cell for the points. Defaults to ``None``.
            pbc (bool or 3-tuple of bool, optional): whether to apply periodic boundary conditions to all or each of
            the three cell vector directions. Defaults to ``False``.
            self_interaction (bool, optional): whether to include self edges for points. Defaults to ``False``. Note
            that edges between the same atom in different periodic images are still included. (See
            ``strict_self_interaction`` to control this behaviour.)
            strict_self_interaction (bool): Whether to include *any* self interaction edges in the graph, even if the
            two instances of the atom are in different periodic images. Defaults to True, should be True for most
            applications.
            **kwargs (optional): other information to pass to the ``torch_geometric.data.Data`` constructor for
            inclusion in the object. Keys listed in ``AtomicDataDict.*_KEY` will be treated specially.
        """
        if pos is None or r_max is None:
            raise ValueError("pos and r_max must be given.")

        if pbc is None:
            if cell is not None:
                raise ValueError(
                    "A cell was provided, but pbc weren't. Please explicitly probide PBC."
                )
            # there are no PBC if cell and pbc are not provided
            pbc = False

        if isinstance(pbc, bool):
            pbc = (pbc,) * 3
        else:
            assert len(pbc) == 3

        pos = torch.as_tensor(pos, dtype=torch.get_default_dtype())

        edge_index, edge_cell_shift, cell = neighbor_list_and_relative_vec(
            pos=pos,
            r_max=r_max,
            self_interaction=self_interaction,
            strict_self_interaction=strict_self_interaction,
            cell=cell,
            pbc=pbc,
        )

        # Make torch tensors for data:
        if cell is not None:
            kwargs[AtomicDataDict.CELL_KEY] = cell.view(3, 3)
            kwargs[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = edge_cell_shift
        if pbc is not None:
            kwargs[AtomicDataDict.PBC_KEY] = torch.as_tensor(
                pbc, dtype=torch.bool
            ).view(3)

        return cls(edge_index=edge_index, pos=torch.as_tensor(pos), **kwargs)

    @classmethod
    def from_ase(cls, atoms, r_max, **kwargs):
        """Build a ``AtomicData`` from an ``ase.Atoms`` object.

        Respects ``atoms``'s ``pbc`` and ``cell``.

        Automatically recognize force, energy (overridden by free energy tag)
        get_atomic_numbers() will be stored as the atomic_numbers attributes


        Args:
            atoms (ase.Atoms): the input.
            r_max (float): neighbor cutoff radius.
            features (torch.Tensor shape [N, M], optional): per-atom M-dimensional feature vectors. If ``None`` (the
             default), uses a one-hot encoding of the species present in ``atoms``.
            **kwargs (optional): other arguments for the ``AtomicData`` constructor.

        Returns:
            A ``AtomicData``.
        """
        add_fields = {}
        if atoms.calc is not None:
            if isinstance(
                atoms.calc, (SinglePointCalculator, SinglePointDFTCalculator)
            ):
                add_fields = deepcopy(atoms.calc.results)
                if "forces" in add_fields:
                    add_fields.pop("forces")
                    add_fields[AtomicDataDict.FORCE_KEY] = atoms.get_forces()

                if "free_energy" in add_fields and "energy" not in add_fields:
                    add_fields[AtomicDataDict.TOTAL_ENERGY_KEY] = add_fields.pop(
                        "free_energy"
                    )
                elif "energy" in add_fields:
                    add_fields[AtomicDataDict.TOTAL_ENERGY_KEY] = add_fields.pop(
                        "energy"
                    )

        elif "forces" in atoms.arrays:
            add_fields[AtomicDataDict.FORCE_KEY] = atoms.arrays["forces"]
        elif "force" in atoms.arrays:
            add_fields[AtomicDataDict.FORCE_KEY] = atoms.arrays["force"]

        add_fields[AtomicDataDict.ATOMIC_NUMBERS_KEY] = atoms.get_atomic_numbers()

        return cls.from_points(
            pos=atoms.positions,
            r_max=r_max,
            cell=atoms.get_cell(),
            pbc=atoms.pbc,
            **kwargs,
            **add_fields,
        )

    def to_ase(self) -> Union[List[ase.Atoms], ase.Atoms]:
        """Build a (list of) ``ase.Atoms`` object(s) from an ``AtomicData`` object.

        For each unique batch number provided in ``AtomicDataDict.BATCH_KEY``,
        an ``ase.Atoms`` object is created. If ``AtomicDataDict.BATCH_KEY`` does not
        exist in self, a single ``ase.Atoms`` object is created.

        Returns:
            A list of ``ase.Atoms`` objects if ``AtomicDataDict.BATCH_KEY`` is in self
            and is not None. Otherwise, a single ``ase.Atoms`` object is returned.
        """
        positions = self.pos
        if positions.device != torch.device("cpu"):
            raise TypeError(
                "Explicitly move this `AtomicData` to CPU using `.to()` before calling `to_ase()`."
            )
        atomic_nums = self.atomic_numbers
        pbc = getattr(self, AtomicDataDict.PBC_KEY, None)
        cell = getattr(self, AtomicDataDict.CELL_KEY, None)
        batch = getattr(self, AtomicDataDict.BATCH_KEY, None)

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
            else:
                mask = slice(None)
            mol = ase.Atoms(
                numbers=atomic_nums[mask],
                positions=positions[mask],
                cell=cell[batch_idx] if cell is not None else None,
                pbc=pbc[batch_idx] if pbc is not None else None,
            )
            batch_atoms.append(mol)

        if batch is not None:
            return batch_atoms
        else:
            assert len(batch_atoms) == 1
            return batch_atoms[0]

    def get_edge_vectors(data: Data) -> torch.Tensor:
        data = AtomicDataDict.with_edge_vectors(AtomicData.to_AtomicDataDict(data))
        return data[AtomicDataDict.EDGE_VECTORS_KEY]

    @staticmethod
    def to_AtomicDataDict(
        data: Union[Data, Mapping], exclude_keys=tuple()
    ) -> AtomicDataDict.Type:
        if isinstance(data, Data):
            keys = data.keys
        elif isinstance(data, Mapping):
            keys = data.keys()
        else:
            raise ValueError(f"Invalid data `{repr(data)}`")

        return {
            k: data[k]
            for k in keys
            if (
                k not in exclude_keys
                and data[k] is not None
                and isinstance(data[k], torch.Tensor)
            )
        }

    @classmethod
    def from_AtomicDataDict(cls, data: AtomicDataDict.Type):
        return cls(**data)

    @property
    def irreps(self):
        return self.__irreps__

    def __cat_dim__(self, key, value):
        if key in (
            AtomicDataDict.CELL_KEY,
            AtomicDataDict.PBC_KEY,
            AtomicDataDict.TOTAL_ENERGY_KEY,
        ):
            # the cell and PBC are graph-level properties and so need a new batch dimension
            return None
        else:
            return super().__cat_dim__(key, value)

    def without_nodes(self, which_nodes):
        """Return a copy of ``self`` with ``which_nodes`` removed.
        The returned object may share references to some underlying data tensors with ``self``.
        Args:
            which_nodes (index tensor or boolean mask)
        Returns:
            A new data object.
        """
        which_nodes = torch.as_tensor(which_nodes)
        if which_nodes.dtype == torch.bool:
            mask = ~which_nodes
        else:
            mask = torch.ones(self.num_nodes, dtype=torch.bool)
            mask[which_nodes] = False
        assert mask.shape == (self.num_nodes,)
        n_keeping = mask.sum()

        # Only keep edges where both from and to are kept
        edge_mask = mask[self.edge_index[0]] & mask[self.edge_index[1]]
        # Create an index mapping:
        new_index = torch.full((self.num_nodes,), -1, dtype=torch.long)
        new_index[mask] = torch.arange(n_keeping, dtype=torch.long)

        new_dict = {}
        for k in self.keys:
            if k == AtomicDataDict.EDGE_INDEX_KEY:
                new_dict[AtomicDataDict.EDGE_INDEX_KEY] = new_index[
                    self.edge_index[:, edge_mask]
                ]
            elif k == AtomicDataDict.EDGE_CELL_SHIFT_KEY:
                new_dict[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = self.edge_cell_shift[
                    edge_mask
                ]
            elif k == AtomicDataDict.CELL_KEY:
                new_dict[k] = self[k]
            else:
                if isinstance(self[k], torch.Tensor) and len(self[k]) == self.num_nodes:
                    new_dict[k] = self[k][mask]
                else:
                    new_dict[k] = self[k]

        new_dict["irreps"] = self.__irreps__

        return type(self)(**new_dict)


def neighbor_list_and_relative_vec(
    pos,
    r_max,
    self_interaction=False,
    strict_self_interaction=True,
    cell=None,
    pbc=False,
):
    """Create neighbor list and neighbor vectors based on radial cutoff.

    Create neighbor list (``edge_index``) and relative vectors
    (``edge_attr``) based on radial cutoff.

    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).

    Thus, ``edge_index`` has the same convention as the relative vectors:
    :math:`\\vec{r}_{source, target}`

    If the input positions are a tensor with ``requires_grad == True``,
    the output displacement vectors will be correctly attached to the inputs
    for autograd.

    All outputs are Tensors on the same device as ``pos``; this allows future
    optimization of the neighbor list on the GPU.

    Args:
        pos (shape [N, 3]): Positional coordinate; Tensor or numpy array. If Tensor, must be on CPU.
        r_max (float): Radial cutoff distance for neighbor finding.
        cell (numpy shape [3, 3]): Cell for periodic boundary conditions. Ignored if ``pbc == False``.
        pbc (bool or 3-tuple of bool): Whether the system is periodic in each of the three cell dimensions.
        self_interaction (bool): Whether or not to include same periodic image self-edges in the neighbor list.
        strict_self_interaction (bool): Whether to include *any* self interaction edges in the graph, even if the two
            instances of the atom are in different periodic images. Defaults to True, should be True for most applications.

    Returns:
        edge_index (torch.tensor shape [2, num_edges]): List of edges.
        edge_cell_shift (torch.tensor shape [num_edges, 3]): Relative cell shift
            vectors. Returned only if cell is not None.
        cell (torch.Tensor [3, 3]): the cell as a tensor on the correct device.
            Returned only if cell is not None.
    """
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3

    # Either the position or the cell may be on the GPU as tensors
    if isinstance(pos, torch.Tensor):
        temp_pos = pos.detach().cpu().numpy()
        out_device = pos.device
        out_dtype = pos.dtype
    else:
        temp_pos = np.asarray(pos)
        out_device = torch.device("cpu")
        out_dtype = torch.get_default_dtype()

    # Right now, GPU tensors require a round trip
    if out_device.type != "cpu":
        warnings.warn(
            "Currently, neighborlists require a round trip to the CPU. Please pass CPU tensors if possible."
        )

    # Get a cell on the CPU no matter what
    if isinstance(cell, torch.Tensor):
        temp_cell = cell.detach().cpu().numpy()
        cell_tensor = cell.to(device=out_device, dtype=out_dtype)
    elif cell is not None:
        temp_cell = np.asarray(cell)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)
    else:
        # ASE will "complete" this correctly.
        temp_cell = np.zeros((3, 3), dtype=temp_pos.dtype)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)

    # ASE dependent part
    temp_cell = ase.geometry.complete_cell(temp_cell)

    first_idex, second_idex, shifts = ase.neighborlist.primitive_neighbor_list(
        "ijS",
        pbc,
        temp_cell,
        temp_pos,
        cutoff=float(r_max),
        self_interaction=strict_self_interaction,  # we want edges from atom to itself in different periodic images!
        use_scaled_positions=False,
    )

    # Eliminate true self-edges that don't cross periodic boundaries
    if not self_interaction:
        bad_edge = first_idex == second_idex
        bad_edge &= np.all(shifts == 0, axis=1)
        keep_edge = ~bad_edge
        if not np.any(keep_edge):
            raise ValueError(
                "After eliminating self edges, no edges remain in this system."
            )
        first_idex = first_idex[keep_edge]
        second_idex = second_idex[keep_edge]
        shifts = shifts[keep_edge]

    # Build output:
    edge_index = torch.vstack(
        (torch.LongTensor(first_idex), torch.LongTensor(second_idex))
    ).to(device=out_device)

    shifts = torch.as_tensor(
        shifts,
        dtype=out_dtype,
        device=out_device,
    )
    return edge_index, shifts, cell_tensor
