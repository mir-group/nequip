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
            return loss.sum()


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
        # average over xyz
        per_atom_loss = self.func(pred[key], ref[key])
        per_atom_loss = per_atom_loss.mean(dim=-1, keepdim=True)

        # zero the nan entries
        not_nan = ~torch.isnan(per_atom_loss)
        per_atom_loss = torch.nan_to_num(per_atom_loss, nan=0.0)

        species_index = pred[AtomicDataDict.SPECIES_INDEX_KEY]
        unique_indices, inverse_species_index = torch.unique(
            species_index, return_inverse=True
        )

        per_species_loss = scatter(
            per_atom_loss, inverse_species_index, reduce="sum", dim=0
        )

        # count the number of species, excluding the nan entry
        ones = torch.ones_like(per_atom_loss, dtype=torch.int8) * not_nan
        weight_species = 1.0 / scatter(ones, inverse_species_index, reduce="sum", dim=0)

        # the species that have all entry with nan value will be nan
        # set it to zero
        not_inf = ~torch.isinf(weight_species)
        weight_species = torch.nan_to_num(weight_species * not_inf, nan=0.0)

        sum = (per_species_loss * weight_species).sum()

        if mean:
            return sum / torch.sum(not_inf)
        else:
            return sum / torch.sum(not_inf) * per_atom_loss.size[0]


def find_loss_function(name: str, params):

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
