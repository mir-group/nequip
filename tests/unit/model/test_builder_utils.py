import pytest
import tempfile

import numpy as np
import torch

import ase.io

from nequip.data import AtomicDataDict, ASEDataset
from nequip.data.transforms import TypeMapper

from nequip.model.builder_utils import add_avg_num_neighbors, add_avg_num_atoms


@pytest.mark.parametrize("r_max", [3.0, 2.0, 1.1])
@pytest.mark.parametrize("subset", [False, True])
@pytest.mark.parametrize("to_test", ["neighbors", "atoms"])
def test_avg_num(molecules, temp_data, r_max, subset, to_test):
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
            AtomicData_options={"r_max": r_max},
            ase_args=dict(format="extxyz"),
            type_mapper=TypeMapper(chemical_symbol_to_type={"H": 0, "C": 1, "O": 2}),
        )

    if subset:
        old_nequip_dataset = nequip_dataset  # noqa
        nequip_dataset = nequip_dataset.index_select(
            torch.randperm(len(nequip_dataset))[: len(nequip_dataset) // 2]
        )

    func = {"neighbors": add_avg_num_neighbors, "atoms": add_avg_num_atoms}[to_test]

    # test basic options
    annkey = f"avg_num_{to_test}"
    config = {annkey: 3}
    func(config, initialize=False, dataset=None)
    assert config[annkey] == 3.0  # nothing should happen
    assert isinstance(config[annkey], float)

    config = {annkey: 3}
    # dont need dataset if config isn't auto
    func(config, initialize=False, dataset=None)
    with pytest.raises(ValueError):
        # need if it is
        config = {annkey: "auto"}
        func(config, initialize=True, dataset=None)

    # compute dumb truth
    if to_test == "neighbors":
        num_neigh = []
        for i in range(len(nequip_dataset)):
            frame = nequip_dataset[i]
            num_neigh.append(
                torch.bincount(
                    frame[AtomicDataDict.EDGE_INDEX_KEY][0],
                    minlength=len(frame[AtomicDataDict.POSITIONS_KEY]),
                ).float()
            )
        avg_num_truth = torch.mean(torch.cat(num_neigh, dim=0))
    elif to_test == "atoms":
        num_atoms = []
        for i in range(len(nequip_dataset)):
            frame = nequip_dataset[i]
            num_atoms.append(frame.num_nodes)
        avg_num_truth = np.mean(num_atoms)
    else:
        raise AssertionError

    # compare
    config = {annkey: "auto"}
    func(config, initialize=True, dataset=nequip_dataset)
    assert config[annkey] == avg_num_truth
