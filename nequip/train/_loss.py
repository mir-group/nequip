import logging

import torch.nn
from torch_scatter import scatter

from nequip.data import AtomicDataDict
from nequip.utils import instantiate_from_cls_name


class SimpleLoss:
    """wrapper to compute weighted loss function
    if atomic_weight_on is True, the loss function will search for
    AtomicDataDict.WEIGHTS_KEY+key in the reference data.

    Args:

    func_name (str): any loss function defined in torch.nn that
        takes "reduction=none" as init argument, uses prediction tensor,
        and reference tensor for its call functions, and outputs a vector
        with the same shape as pred/ref
    params (str): arguments needed to initialize the function above

    Return:

    if mean is True, return a scalar; else return the error matrix of each entry
    """

    def __init__(self, func_name: str, params: dict = {}):
        func, _ = instantiate_from_cls_name(
            torch.nn,
            class_name=func_name,
            prefix="",
            positional_args=dict(reduction="none"),
            optional_args=params,
            all_args={},
        )
        self.func = func

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):

        # zero the nan entries
        not_nan = ~torch.isnan(ref[key])
        loss = torch.nan_to_num(self.func(pred[key], ref[key]), nan=0.0)
        if mean:
            return loss.sum() / not_nan.sum()
        else:
            return loss


class PerSpeciesLoss(SimpleLoss):
    """Compute loss for each species and average among the same species
    before summing them up.

    Args same as SimpleLoss
    """

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        if not mean:
            raise NotImplementedError("Cannot handle this yet")

        # average over xyz
        per_atom_loss = self.func(pred[key], ref[key])
        per_atom_loss = per_atom_loss.mean(dim=-1, keepdim=True)

        # find out what is nan
        not_nan = torch.reshape(~torch.isnan(per_atom_loss), (-1, ))

        # offset species index by 1 to use 0 for nan
        species_index = pred[AtomicDataDict.SPECIES_INDEX_KEY]+1
        unique_indices, inverse_species_index, species_count = torch.unique(
            species_index*not_nan, return_inverse=True, return_counts=True,
        )
        weight_species = 1.0 / species_count

        per_species_loss = torch.reshape(scatter(
            per_atom_loss, inverse_species_index, reduce="sum", dim=0
        ), (-1,))

        # zero the nan entries
        not_nan_count = torch.sum(~torch.isnan(per_species_loss))
        per_species_loss = torch.nan_to_num(per_species_loss, nan=0.0)

        sum = (per_species_loss * weight_species).sum()

        return sum / not_nan_count


def find_loss_function(name: str, params):
    """
    Search for loss functions in this module

    If the name starts with PerSpecies, return the PerSpeciesLoss instance
    """

    wrapper_list = dict(
        PerSpecies=PerSpeciesLoss,
    )

    if isinstance(name, str):
        for key in wrapper_list:
            if name.startswith(key):
                logging.debug(f"create loss instance {wrapper_list[key]}")
                return wrapper_list[key](name[len(key) :], params)

        return SimpleLoss(name, params)
    elif callable(name):
        return name
    else:
        raise NotImplementedError(f"{name} Loss is not implemented")
