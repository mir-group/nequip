from typing import Optional

import torch

from nequip.utils import Config
from nequip.data import AtomicDataset, AtomicDataDict


def _add_avg_num_neighbors_helper(data):
    counts = torch.unique(
        data[AtomicDataDict.EDGE_INDEX_KEY][0],
        sorted=True,
        return_counts=True,
    )[1]
    # in case the cutoff is small and some nodes have no neighbors,
    # we need to pad `counts` up to the right length
    counts = torch.nn.functional.pad(
        counts, pad=(0, len(data[AtomicDataDict.POSITIONS_KEY]) - len(counts))
    )
    return (counts, "node")


def add_avg_num_neighbors(
    config: Config,
    initialize: bool,
    dataset: Optional[AtomicDataset] = None,
) -> Optional[float]:
    # Compute avg_num_neighbors
    annkey: str = "avg_num_neighbors"
    var_nn_key: str = "var_num_neighbors"
    ann = config.get(annkey, "auto")
    var_nn = None
    if ann == "auto":
        if not initialize:
            raise ValueError("avg_num_neighbors = auto but initialize is False")
        if dataset is None:
            raise ValueError(
                "When avg_num_neighbors = auto, the dataset is required to build+initialize a model"
            )
        ann, var_nn = dataset.statistics(
            fields=[_add_avg_num_neighbors_helper],
            modes=["mean_std"],
            stride=config.get("dataset_statistics_stride", 1),
        )[0]
        ann = ann.item()
        var_nn = var_nn.item()

    # make sure its valid
    if ann is not None:
        ann = float(ann)
    config[annkey] = ann
    if var_nn is not None:
        config[var_nn_key] = var_nn
    return ann
