# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from torch.utils.data import Dataset
from typing import Dict, Union


class SubsetByRandomSlice(torch.utils.data.Subset):
    """Subset of dataset by slicing a random permutation of the dataset.

    Args:
        dataset (Dataset): ``torch.utils.data.Dataset`` to get subset of
        start (int): starting index for the slice
        length (int): number of samples to slice from ``start``
        seed (int): seed for reproducibility of the random permutation of indices
    """

    def __init__(
        self,
        dataset: Dataset,
        start: int,
        length: int,
        seed: int,
    ):
        data_len = len(dataset)
        assert (
            length <= data_len
        ), f"Unable to get a subset (length {length}) larger than the size of the dataset (length {data_len}) provided"
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)

        indices = indices[slice(start, start + length)]
        super().__init__(dataset, indices)


def RandomSplitAndIndexDataset(
    dataset: torch.utils.data.Dataset,
    split_dict: Dict[str, Union[int, float]],
    dataset_key: str,
    seed: int,
) -> torch.utils.data.Dataset:
    """
    Args:
        dataset (Dataset): the base dataset that is to be split
        split_dict (Dict): dictionary with signature ``{name_of_subset: num_data/frac_data}`` where ``num_data`` must sum up to the size of the given dataset or ``frac_data`` must sum up to 1
        dataset_key (str): name of the data subset to return
        seed (int)       : seed for reproducible splits
    """
    # make a new generator from the seed every time a split is done -- reproducible splits as long as seed is the same
    generator = torch.Generator().manual_seed(seed)
    # API based on dicts (instead of lists and indices) makes it easier to keep track of what each dataset entry is
    subset_names = list(split_dict.keys())
    lengths = [split_dict[name] for name in subset_names]
    # torch.utils.data.random_split will error out if the splits don't make sense, e.g. don't sum up to  num_data or 1
    # => no need to do safety checks on our part (though the error only appears when a split is first attempted)
    splits = torch.utils.data.random_split(dataset, lengths, generator=generator)
    return splits[subset_names.index(dataset_key)]
