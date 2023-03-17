import numpy as np
from os.path import dirname, basename, abspath
from typing import Dict, Any, List, Optional


from .. import AtomicDataDict, _LONG_FIELDS, _NODE_FIELDS, _GRAPH_FIELDS
from ..transforms import TypeMapper
from ._base_datasets import AtomicInMemoryDataset


class NpzDataset(AtomicInMemoryDataset):
    """Load data from an npz file.

    To avoid loading unneeded data, keys are ignored by default unless they are in ``key_mapping``, ``include_keys``,
    or ``npz_fixed_fields_keys``.

    Args:
        key_mapping (Dict[str, str]): mapping of npz keys to ``AtomicData`` keys. Optional
        include_keys (list): the attributes to be processed and stored. Optional
        npz_fixed_field_keys: the attributes that only have one instance but apply to all frames. Optional
            Note that the mapped keys (as determined by the _values_ in ``key_mapping``) should be used in
            ``npz_fixed_field_keys``, not the original npz keys from before mapping. If an npz key is not
            present in ``key_mapping``, it is mapped to itself, and this point is not relevant.

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
    graph_fields:
      - user_label1
    node_fields:
      - user_label2
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
        AtomicData_options: Dict[str, Any] = {},
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
            AtomicData_options=AtomicData_options,
            include_frames=include_frames,
            type_mapper=type_mapper,
        )

    @property
    def raw_file_names(self):
        return [basename(self.file_name)]

    @property
    def raw_dir(self):
        return dirname(abspath(self.file_name))

    def get_data(self):

        data = np.load(self.raw_dir + "/" + self.raw_file_names[0], allow_pickle=True)

        # only the keys explicitly mentioned in the yaml file will be parsed
        keys = set(list(self.key_mapping.keys()))
        keys.update(self.npz_fixed_field_keys)
        keys.update(self.include_keys)
        keys = keys.intersection(set(list(data.keys())))

        mapped = {self.key_mapping.get(k, k): data[k] for k in keys}

        for intkey in _LONG_FIELDS:
            if intkey in mapped:
                mapped[intkey] = mapped[intkey].astype(np.int64)

        fields = {k: v for k, v in mapped.items() if k not in self.npz_fixed_field_keys}
        num_examples, num_atoms, n_dim = fields[AtomicDataDict.POSITIONS_KEY].shape
        assert n_dim == 3

        # now we replicate and add the fixed fields:
        for fixed_field in self.npz_fixed_field_keys:
            orig = mapped[fixed_field]
            if fixed_field in _NODE_FIELDS:
                assert orig.ndim >= 1  # [n_atom, feature_dims]
                assert orig.shape[0] == num_atoms
                replicated = np.expand_dims(orig, 0)
                replicated = np.tile(
                    replicated,
                    (num_examples,) + (1,) * len(replicated.shape[1:]),
                )  # [n_example, n_atom, feature_dims]
            elif fixed_field in _GRAPH_FIELDS:
                # orig is [feature_dims]
                replicated = np.expand_dims(orig, 0)
                replicated = np.tile(
                    replicated,
                    (num_examples,) + (1,) * len(replicated.shape[1:]),
                )  # [n_example, feature_dims]
            else:
                raise KeyError(
                    f"npz_fixed_field_keys contains `{fixed_field}`, but it isn't registered as a node or graph field"
                )
            fields[fixed_field] = replicated
        return fields
