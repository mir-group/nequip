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
        self.has_nan = params.get("has_nan", False)
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

        per_atom_loss = self.func(pred[key], ref[key])
        has_nan = self.has_nan and torch.isnan(per_atom_loss.mean())

        if has_nan:
            not_nan = (per_atom_loss == per_atom_loss).int()
            per_atom_loss = torch.nan_to_num(per_atom_loss, nan=0.0)

        reduce_dims = tuple(i + 1 for i in range(len(per_atom_loss.shape) - 1))

        if has_nan:
            if len(reduce_dims)>0:
                per_atom_loss = per_atom_loss.sum(dim=reduce_dims)

            spe_idx = pred[AtomicDataDict.SPECIES_INDEX_KEY]
            per_species_loss = scatter(per_atom_loss, spe_idx, reduce="sum", dim=0)

            N = scatter(not_nan, spe_idx, reduce="sum", dim=0)
            N = N.sum(reduce_dims)
            N = 1.0 / N
            N_species = ((N == N).int()).sum()

            return (per_species_loss * N).sum() / N_species

        else:

            if len(reduce_dims)>0:
                per_atom_loss = per_atom_loss.mean(dim=reduce_dims)

            # offset species index by 1 to use 0 for nan
            spe_idx = pred[AtomicDataDict.SPECIES_INDEX_KEY]
            _, inverse_species_index = torch.unique(spe_idx, return_inverse=True)

            per_species_loss = scatter(per_atom_loss, inverse_species_index, reduce="mean", dim=0)

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
