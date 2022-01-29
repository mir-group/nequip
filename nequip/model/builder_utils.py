from typing import Optional

import torch

from nequip.utils import Config
from nequip.data import AtomicDataset, AtomicDataDict


def add_avg_num_neighbors(
    config: Config,
    initialize: bool,
    dataset: Optional[AtomicDataset] = None,
) -> Optional[float]:
    # Compute avg_num_neighbors
    annkey: str = "avg_num_neighbors"
    ann = config.get(annkey, None)
    if ann == "auto":
        if not initialize:
            raise ValueError("avg_num_neighbors = auto but initialize is False")
        if dataset is None:
            raise ValueError(
                "When avg_num_neighbors = auto, the dataset is required to build+initialize a model"
            )
        ann = dataset.statistics(
            fields=[
                lambda data: (
                    torch.bincount(
                        data[AtomicDataDict.EDGE_INDEX_KEY][0],
                        # make sure we have the right number of counts even if some nodes have no neighbors
                        minlength=len(data[AtomicDataDict.POSITIONS_KEY]),
                    ),
                    "node",
                )
            ],
            modes=["mean_std"],
            stride=config.get("dataset_statistics_stride", 1),
        )[0][0].item()

    # make sure its valid
    if ann is not None:
        ann = float(ann)
    config[annkey] = ann
    return ann
