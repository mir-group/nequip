# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Any, Dict, List, Optional, Union

import torch

from e3nn.o3._irreps import Irreps
from e3nn.o3._spherical_harmonics import SphericalHarmonics

from nequip.data import AtomicDataDict
from nequip.data._key_registry import get_field_type
from .._graph_mixin import GraphModuleMixin


class AppendVectorFieldEmbed(GraphModuleMixin, torch.nn.Module):
    """Append embedded node or graph vector fields to node features.

    Args:
        vector_fields: names of registered vector fields to embed and append.
        field_sh_irreps: irreps used by `SphericalHarmonics` for each vector field.
        append_to_node_attrs: if True, keep `node_attrs` equal to appended `node_features`.
        irreps_in: input irreps dictionary passed to `GraphModuleMixin`.
    """

    def __init__(
        self,
        vector_fields: List[str],
        field_sh_irreps: Union[int, str, Irreps],
        append_to_node_attrs: bool = True,
        irreps_in: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        irreps_in = {} if irreps_in is None else dict(irreps_in)
        self.append_to_node_attrs = append_to_node_attrs

        assert AtomicDataDict.NODE_FEATURES_KEY in irreps_in, (
            f"`{AtomicDataDict.NODE_FEATURES_KEY}` must be present in `irreps_in`"
        )
        if self.append_to_node_attrs:
            assert AtomicDataDict.NODE_ATTRS_KEY in irreps_in, (
                f"`{AtomicDataDict.NODE_ATTRS_KEY}` must be present in `irreps_in` when `append_to_node_attrs=True`"
            )

        self.vector_fields = list(vector_fields)
        self.field_kinds = self._validate_fields(self.vector_fields)

        required_vector_irreps = Irreps("1e")
        for field in self.vector_fields:
            if field in irreps_in:
                assert irreps_in[field] == required_vector_irreps, (
                    f"`{field}` must have irreps {required_vector_irreps}, but got {irreps_in[field]}"
                )
            else:
                irreps_in[field] = required_vector_irreps

        if isinstance(field_sh_irreps, int):
            self.field_sh_irreps = Irreps.spherical_harmonics(lmax=field_sh_irreps)
        else:
            self.field_sh_irreps = Irreps(field_sh_irreps)
        # don't normalize SH for field vectors; this gives solid harmonics
        self.sh = SphericalHarmonics(
            self.field_sh_irreps,
            normalize=False,
            normalization="component",
        )

        self.extra_irreps = Irreps()
        for _ in self.vector_fields:
            self.extra_irreps += self.field_sh_irreps

        irreps_out = {
            AtomicDataDict.NODE_FEATURES_KEY: (
                irreps_in[AtomicDataDict.NODE_FEATURES_KEY] + self.extra_irreps
            )
        }
        if self.append_to_node_attrs:
            irreps_out[AtomicDataDict.NODE_ATTRS_KEY] = (
                irreps_in[AtomicDataDict.NODE_ATTRS_KEY] + self.extra_irreps
            )
        required_irreps_in = [AtomicDataDict.NODE_FEATURES_KEY]
        if self.append_to_node_attrs:
            required_irreps_in.append(AtomicDataDict.NODE_ATTRS_KEY)
        required_irreps_in.extend(self.vector_fields)

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=required_irreps_in,
            irreps_out=irreps_out,
        )

        self.model_dtype = torch.get_default_dtype()

    @staticmethod
    def _validate_fields(vector_fields: List[str]) -> Dict[str, str]:
        assert len(vector_fields) > 0, "`vector_fields` cannot be empty"
        duplicate_fields = [
            field for i, field in enumerate(vector_fields) if field in vector_fields[:i]
        ]
        assert len(duplicate_fields) == 0, (
            f"duplicate fields in `vector_fields`: {duplicate_fields}"
        )

        field_kinds = {}
        for field in vector_fields:
            field_kind = get_field_type(field, error_on_unregistered=True)
            assert field_kind in ("graph", "node"), (
                f"`{field}` has field type `{field_kind}` but only graph/node fields can be appended"
            )
            field_kinds[field] = field_kind
        return field_kinds

    def _field_to_per_node(
        self,
        data: AtomicDataDict.Type,
        field: str,
        num_nodes: int,
    ) -> torch.Tensor:
        value = data[field].view(-1, 3)
        field_kind = self.field_kinds[field]
        # short-circuit of node case
        if field_kind == "node":
            return value

        # (num_graph, 3) -> (num_nodes, 3)
        if AtomicDataDict.BATCH_KEY in data:
            batch = data[AtomicDataDict.BATCH_KEY].view(-1)
            return torch.index_select(value, 0, batch)
        # unbatched case -> all nodes get same value
        return value.expand(num_nodes, 3)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        node_features = data[AtomicDataDict.NODE_FEATURES_KEY]

        embedded_fields = []
        for field in self.vector_fields:
            per_node_vector = self._field_to_per_node(
                data=data,
                field=field,
                num_nodes=node_features.size(0),
            )
            embedded_fields.append(self.sh(per_node_vector).to(dtype=self.model_dtype))

        # build the concatenation input list explicitly to satisfy TorchScript
        cat_inputs = [node_features]
        for embedded in embedded_fields:
            cat_inputs.append(embedded)
        node_features = torch.cat(cat_inputs, dim=1)
        data[AtomicDataDict.NODE_FEATURES_KEY] = node_features

        if self.append_to_node_attrs:
            data[AtomicDataDict.NODE_ATTRS_KEY] = node_features

        return data
