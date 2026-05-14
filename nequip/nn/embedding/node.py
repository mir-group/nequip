# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from dataclasses import dataclass
from math import sqrt
import torch

from e3nn.o3._irreps import Irreps

from nequip.data import AtomicDataDict
from nequip.data._key_registry import _GRAPH_FIELDS
from .._graph_mixin import GraphModuleMixin

from typing import Optional, Final, List, Dict, Any


@dataclass(frozen=True)
class CategoricalGraphFieldEmbedSpec:
    field: str
    num_features: int
    min: int
    max: int
    init: Optional[str] = None

    @classmethod
    def from_dict(cls, field_embed: Dict[str, Any]) -> "CategoricalGraphFieldEmbedSpec":
        required_keys: Final[List[str]] = ["field", "num_features", "min", "max"]
        missing_keys = [key for key in required_keys if key not in field_embed]
        assert len(missing_keys) == 0, (
            f"missing keys {missing_keys} in `categorical_graph_field_embed` entry; required keys are {required_keys}."
        )
        return cls(
            field=str(field_embed["field"]),
            num_features=int(field_embed["num_features"]),
            min=int(field_embed["min"]),
            max=int(field_embed["max"]),
            init=field_embed.get("init", None),
        )


class NodeTypeEmbed(GraphModuleMixin, torch.nn.Module):
    """Generates node type embeddings.

    Args:
        type_names (List[str]): list of type names
        num_features (int): embedding dimension
        type_embed_init (str): embedding initialization mode for atom type embeddings.
            One of ``"uniform"``, ``"zero"``, ``"near_zero"``, or ``None`` (default, keep PyTorch behavior).
        set_features (bool): ``node_features`` will be set in addition to ``node_attrs`` if ``True`` (default)
        categorical_graph_field_embed: list of dicts, each dict having keys ``field``, ``num_features``, ``min``, ``max``, and optional ``init``.
            ``field`` must correspond to a registered graph data field.
            The data dict for the field must be populated by an integer quantity that lies between ``min`` and ``max``.
    """

    num_types: int
    set_features: bool
    type_embed_init: Optional[str]

    def __init__(
        self,
        type_names: List[str],
        num_features: int,
        type_embed_init: Optional[str] = None,
        set_features: bool = True,
        categorical_graph_field_embed: Optional[List[Dict[str, Any]]] = None,
        irreps_in: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # normalize optional inputs to avoid shared mutable defaults
        irreps_in = {} if irreps_in is None else dict(irreps_in)
        # === bookkeeping ===
        self.num_types = len(type_names)
        self.set_features = set_features
        self.type_embed_init = type_embed_init

        # === type embedding module ===
        self.embed_module = torch.nn.Embedding(
            num_embeddings=self.num_types,
            embedding_dim=num_features,
        )
        self._init_embedding(self.embed_module, init=self.type_embed_init)

        # === categorical graph field embedding ===
        total_features = num_features
        self.categorical_graph_field_embed_modules = torch.nn.ModuleDict()
        self.categorical_graph_field_embed_shifts = {}
        self.do_categorical_graph_field_embed = False
        if categorical_graph_field_embed is not None:
            self.do_categorical_graph_field_embed = True
            for field_embed_dict in categorical_graph_field_embed:
                field_embed = CategoricalGraphFieldEmbedSpec.from_dict(field_embed_dict)
                assert field_embed.field in _GRAPH_FIELDS, (
                    f"`{field_embed.field}` is not a graph field, only graph fields should be provided to `categorical_graph_field_embed`."
                )
                assert field_embed.max >= field_embed.min, (
                    f"`max` must be >= `min` for field `{field_embed.field}`."
                )
                field_init = field_embed.init

                # == important inits ==
                embed_module = torch.nn.Embedding(
                    num_embeddings=field_embed.max - field_embed.min + 1,
                    embedding_dim=field_embed.num_features,
                )
                self._init_embedding(embed_module, init=field_init)
                self.categorical_graph_field_embed_modules.update(
                    {field_embed.field: embed_module}
                )
                self.categorical_graph_field_embed_shifts.update(
                    {field_embed.field: field_embed.min}
                )
                # ^ we subtract this quantity to make sure the smallest index is 0

                # == bookkeeping ==
                total_features += field_embed.num_features

                # register `irreps_in` if not already done
                # needed to ensure that the field is propagated into the model
                if field_embed.field not in irreps_in:
                    # categorical, so no irreps
                    irreps_in[field_embed.field] = None

        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(total_features, (0, 1))])}
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    @staticmethod
    def _init_embedding(
        module: torch.nn.Embedding,
        init: Optional[str],
    ) -> None:
        if init is None:
            return
        if init == "uniform":
            torch.nn.init.uniform_(module.weight, -sqrt(3.0), sqrt(3.0))
        elif init == "zero":
            torch.nn.init.zeros_(module.weight)
        elif init == "near_zero":
            torch.nn.init.normal_(module.weight, mean=0.0, std=1e-5)
        else:
            raise ValueError(
                f"unsupported embedding init mode `{init}`. supported modes: ('uniform', 'zero', 'near_zero') or None"
            )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # (num_atoms, 1) -> (num_atoms, num_type_features)
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].view(-1)
        embedding = self.embed_module(atom_types)

        # handle categorical graph field embeddings
        if self.do_categorical_graph_field_embed:
            embeddings = [embedding]
            for field, module in self.categorical_graph_field_embed_modules.items():
                # (num_graph, 1) -> (num_atoms, 1)
                if AtomicDataDict.BATCH_KEY in data:
                    categorical_graph_field = torch.index_select(
                        data[field].view(-1), 0, data[AtomicDataDict.BATCH_KEY].view(-1)
                    )
                else:
                    categorical_graph_field = (
                        data[field].view(-1).expand((atom_types.size(0),))
                    )
                # (num_atoms,) -> (num_atoms, num_extra_features)
                categorical_graph_field_embedding = module(
                    categorical_graph_field
                    - self.categorical_graph_field_embed_shifts[field]
                )
                embeddings.append(categorical_graph_field_embedding)
            embedding = torch.cat(embeddings, dim=1)

        data[AtomicDataDict.NODE_ATTRS_KEY] = embedding
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = embedding
        return data
