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
        loss = self.func(pred[key], ref[key])

        # zero the nan entries
        not_nan = torch.isnan(ref[key])
        has_nan = torch.any(not_nan)
        if has_nan:
            not_nan = ~not_nan
            loss = torch.nan_to_num(loss, nan=0.0)
            if mean:
                return loss.sum() / not_nan.sum()
            else:
                return loss
        else:
            if mean:
                return loss.mean()
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

        # zero the nan entries
        not_nan = torch.isnan(ref[key])
        has_nan = torch.any(not_nan)

        # average over xyz
        per_atom_loss = torch.nan_to_num(self.func(pred[key], ref[key]),
                                         nan=0.0)
        if has_nan:

            per_atom_loss = per_atom_loss.sum(dim=-1, keepdim=True)

            not_nan = ~not_nan
            # offset species index by 1 to use 0 for nan
            spe_idx = pred[AtomicDataDict.SPECIES_INDEX_KEY]

            accumulate_index = (spe_idx+1).reshape((-1,)+(1,)*(len(not_nan.shape)-1))*not_nan

            unique_species, species_weight = torch.unique(
                accumulate_index, return_counts=True,
            )
            unique_species = unique_species[1:]-1
            species_weight = 1./species_weight[1:]
            N_species = len(species_weight)

            per_species_loss = torch.reshape(scatter(
                per_atom_loss, spe_idx, reduce="sum", dim=0
            ), (-1,))

            if len(species_weight)<len(per_species_loss):
                new_weight = torch.zeros(per_species_loss.shape)
                for i, spe in enumerate(unique_species):
                    new_weight[spe] = species_weight[i]
                species_weight = new_weight

            return (per_species_loss * species_weight).sum()/N_species

        else:

            per_atom_loss = per_atom_loss.mean(dim=-1, keepdim=True)

            # offset species index by 1 to use 0 for nan
            spe_idx = pred[AtomicDataDict.SPECIES_INDEX_KEY]
            _, inverse_species_index = torch.unique(
                spe_idx, return_inverse=True
            )

            per_species_loss = torch.reshape(scatter(
                per_atom_loss, inverse_species_index, reduce="mean", dim=0
            ), (-1,))

            return per_species_loss.mean()


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
