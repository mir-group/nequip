import pytest
import tempfile

import torch

import ase.io

from nequip.data import AtomicDataDict, ASEDataset
from nequip.data.transforms import TypeMapper

from nequip.model.builder_utils import add_avg_num_neighbors


@pytest.mark.parametrize("r_max", [3.0, 2.0, 1.1])
def test_avg_num_neighbors(molecules, temp_data, r_max):
    with tempfile.NamedTemporaryFile(suffix=".xyz") as fp:
        for atoms in molecules:
            # Reverse the atoms so the one without neighbors ends up at the end
            # to test the minlength style padding logic
            # this is specific to the current contents and ordering of `molcules`!
            ase.io.write(
                fp.name, ase.Atoms(list(atoms)[::-1]), format="extxyz", append=True
            )
        nequip_dataset = ASEDataset(
            file_name=fp.name,
            root=temp_data,
            extra_fixed_fields={"r_max": r_max},
            ase_args=dict(format="extxyz"),
            type_mapper=TypeMapper(chemical_symbol_to_type={"H": 0, "C": 1, "O": 2}),
        )
    # test basic options
    annkey = "avg_num_neighbors"
    config = {annkey: 3}
    add_avg_num_neighbors(config, initialize=False, dataset=None)
    assert config[annkey] == 3.0  # nothing should happen
    assert isinstance(config[annkey], float)

    config = {annkey: 3}
    # dont need dataset if config isn't auto
    add_avg_num_neighbors(config, initialize=False, dataset=None)
    with pytest.raises(ValueError):
        # need if it is
        config = {annkey: "auto"}
        add_avg_num_neighbors(config, initialize=True, dataset=None)

    # compute dumb truth
    num_neigh = []
    for i in range(len(nequip_dataset)):
        frame = nequip_dataset[i]
        num_neigh.append(
            torch.bincount(
                frame[AtomicDataDict.EDGE_INDEX_KEY][0],
                minlength=len(frame[AtomicDataDict.POSITIONS_KEY]),
            ).float()
        )
    avg_num_neighbor_truth = torch.mean(torch.cat(num_neigh, dim=0))

    # compare
    config = {annkey: "auto"}
    add_avg_num_neighbors(config, initialize=True, dataset=nequip_dataset)
    assert config[annkey] == avg_num_neighbor_truth
