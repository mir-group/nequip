from typing import Optional, Union

import torch

from nequip.utils import Config
from nequip.data import AtomicDataset, AtomicDataDict


def add_avg_num_neighbors(
    config: Config,
    initialize: bool,
    dataset: Optional[AtomicDataset] = None,
    default: Optional[Union[str, float]] = "auto",
) -> Optional[float]:
    # Compute avg_num_neighbors
    annkey: str = "avg_num_neighbors"
    ann = config.get(annkey, default)
    if ann == "auto" and initialize:
        if dataset is None:
            raise ValueError(
                "When avg_num_neighbors = auto, the dataset is required to build+initialize a model"
            )
        ann = dataset.statistics(
            fields=[
                lambda data: (
                    torch.unique(
                        data[AtomicDataDict.EDGE_INDEX_KEY][0], return_counts=True
                    )[1],
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
