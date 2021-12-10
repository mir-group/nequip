import pytest

import torch

from nequip.data import AtomicDataDict

from nequip.model.builder_utils import add_avg_num_neighbors


def test_avg_num_neighbors(nequip_dataset):
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
            torch.bincount(frame[AtomicDataDict.EDGE_INDEX_KEY][0]).float()
        )
    avg_num_neighbor_truth = torch.mean(torch.cat(num_neigh, dim=0))

    # compare
    config = {annkey: "auto"}
    add_avg_num_neighbors(config, initialize=True, dataset=nequip_dataset)
    assert config[annkey] == avg_num_neighbor_truth
