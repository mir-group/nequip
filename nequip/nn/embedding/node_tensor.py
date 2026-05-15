# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Any, Dict, List, Optional

import torch

from e3nn.o3._irreps import Irreps
from e3nn.o3._spherical_harmonics import SphericalHarmonics

from nequip.data import AtomicDataDict
from nequip.data._key_registry import get_field_type
from .._graph_mixin import GraphModuleMixin


class AppendVectorFieldEmbed(GraphModuleMixin, torch.nn.Module):
    """Append embedded node or graph vector fields to node features.

    Each field is embedded via solid harmonics up to ``l_max``.
    The parity of the input vector must be specified per field: ``+1`` for axial vectors
    (pseudovectors, e.g. spin, magnetic field) and ``-1`` for polar vectors (e.g. electric field).

    Args:
        vector_fields: dict mapping field name to its vector parity (+1 or -1).
        l_max: maximum l for the solid harmonic embedding of each field.
        append_to_node_attrs: if True, keep ``node_attrs`` equal to appended ``node_features``.
        irreps_in: input irreps dictionary passed to ``GraphModuleMixin``.
    """

    def __init__(
        self,
        vector_fields: Dict[str, int],
        l_max: int,
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

        assert len(vector_fields) > 0, "`vector_fields` cannot be empty"
        assert all(p in (1, -1) for p in vector_fields.values()), (
            "all parity values in `vector_fields` must be +1 (axial) or -1 (polar)"
        )

        # preserve insertion order for consistent forward indexing
        self.vector_fields: List[str] = list(vector_fields.keys())
        self.field_kinds: Dict[str, str] = self._validate_fields(self.vector_fields)

        # per-field SH modules; e3nn infers irreps_in ("1e" or "1o") from the output irreps
        sh_modules = []
        extra_irreps = Irreps()
        for field, parity in vector_fields.items():
            required_irreps = Irreps("1e" if parity == 1 else "1o")
            if field in irreps_in:
                assert irreps_in[field] == required_irreps, (
                    f"`{field}` must have irreps {required_irreps} for parity {parity:+d}, "
                    f"but got {irreps_in[field]}"
                )
            else:
                irreps_in[field] = required_irreps

            # degree-l SH of a parity-p vector transforms as (l, p**l):
            #   axial  (p=+1): all even  — 0e, 1e, 2e, ...
            #   polar  (p=-1): alternating — 0e, 1o, 2e, ...
            # e3nn validates this and auto-infers irreps_in from these labels
            field_sh_irreps = Irreps([(1, (l, parity**l)) for l in range(l_max + 1)])
            # don't normalize SH for field vectors; this gives solid harmonics
            sh_modules.append(
                SphericalHarmonics(
                    field_sh_irreps, normalize=False, normalization="component"
                )
            )
            extra_irreps += field_sh_irreps

        self.sh_modules = torch.nn.ModuleList(sh_modules)

        irreps_out = {
            AtomicDataDict.NODE_FEATURES_KEY: (
                irreps_in[AtomicDataDict.NODE_FEATURES_KEY] + extra_irreps
            )
        }
        if self.append_to_node_attrs:
            irreps_out[AtomicDataDict.NODE_ATTRS_KEY] = (
                irreps_in[AtomicDataDict.NODE_ATTRS_KEY] + extra_irreps
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

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for field, sh in zip(self.vector_fields, self.sh_modules):
            lines.append(f"  {field}: {sh.irreps_in} -> {sh.irreps_out},")
        lines.append(
            f"  node_features: {self.irreps_in[AtomicDataDict.NODE_FEATURES_KEY]}"
            f" -> {self.irreps_out[AtomicDataDict.NODE_FEATURES_KEY]}"
        )
        lines.append(")")
        return "\n".join(lines)

    @staticmethod
    def _validate_fields(vector_fields: List[str]) -> Dict[str, str]:
        assert len(vector_fields) > 0, "`vector_fields` cannot be empty"
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
        for i, sh in enumerate(self.sh_modules):
            per_node_vector = self._field_to_per_node(
                data=data,
                field=self.vector_fields[i],
                num_nodes=node_features.size(0),
            )
            embedded_fields.append(sh(per_node_vector).to(dtype=self.model_dtype))

        # build the concatenation input list explicitly to satisfy TorchScript
        cat_inputs = [node_features]
        for embedded in embedded_fields:
            cat_inputs.append(embedded)
        node_features = torch.cat(cat_inputs, dim=1)
        data[AtomicDataDict.NODE_FEATURES_KEY] = node_features

        if self.append_to_node_attrs:
            data[AtomicDataDict.NODE_ATTRS_KEY] = node_features

        return data
