from typing import Optional

import torch

from nequip.utils import Config
from nequip.data import AtomicDataset, AtomicDataDict


def add_avg_num_neighbors(
    config: Config, initialize: bool, dataset: Optional[AtomicDataset] = None
) -> Optional[float]:
    # Compute avg_num_neighbors
    annkey: str = "avg_num_neighbors"
    if config.get(annkey, None) == "auto" and initialize:
        if dataset is None:
            raise ValueError(
                "When avg_num_neighbors = auto, the dataset is required to build+initialize a model"
            )
        config[annkey] = dataset.statistics(
            fields=[
                lambda data: (
                    torch.unique(
                        data[AtomicDataDict.EDGE_INDEX_KEY][0], return_counts=True
                    )[1],
                    "node",
                )
            ],
            modes=["mean_std"],
            stride=config.dataset_statistics_stride,
        )[0][0].item()

    # make sure its valid
    ann = config.get(annkey, None)
    if ann is not None:
        config[annkey] = float(config[annkey])
    return config[annkey]
