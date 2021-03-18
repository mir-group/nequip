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

    def __call__(self, pred: dict, ref: dict, key: str, atomic_weight_on: bool):

        loss = self.func(pred[key], ref[key])
        weights_key = AtomicDataDict.WEIGHTS_KEY + key
        if weights_key in ref and atomic_weight_on:
            weights = ref[weights_key]
            loss = (loss * weights).mean() / weights.mean()
        else:
            loss = loss.mean()

        return loss, {key: {key: loss}}  # torch.clone(loss)}}


class PerSpeciesLoss(SimpleLoss):
    """Compute loss for each species and average among the same species
    before summing them up.

    Args same as SimpleLoss
    """

    def __call__(self, pred: dict, ref: dict, key: str, atomic_weight_on: bool):

        per_atom_loss = self.func(pred[key], ref[key])
        per_atom_loss = per_atom_loss.mean(dim=-1, keepdim=True)

        # if there is atomic weights
        weights_key = AtomicDataDict.WEIGHTS_KEY + key
        if weights_key in ref and atomic_weight_on:
            weights = ref[weights_key]
            per_atom_loss = per_atom_loss * weights
        else:
            atomic_weight_on = False

        atomic_number = ref[AtomicDataDict.ATOMIC_NUMBERS_KEY]
        all_species, species_index = torch.unique(atomic_number, return_inverse=True)

        if atomic_weight_on:
            per_species_weight = scatter(weights, species_index, dim=0)
            per_species_loss = scatter(per_atom_loss, species_index, dim=0)
            per_species_loss = per_species_loss / per_species_weight
        else:
            per_species_loss = scatter(
                per_atom_loss, species_index, reduce="mean", dim=0
            )

        total_loss = per_species_loss.mean()
        contrib = {
            int(all_species[i]): {key: per_species_loss[i]}
            for i in range(len(per_species_loss))
        }
        contrib["all"] = {key:total_loss}
        return total_loss, contrib


def find_loss_function(name: str, params: dict = {}):

    wrapper_list = dict(
        PerSpecies=PerSpeciesLoss,
    )
    mae_func = SimpleLoss("L1Loss")

    if isinstance(name, str):
        for key in wrapper_list:
            if name.startswith(key):
                logging.debug(f"create loss instance {wrapper_list[key]}")
                return wrapper_list[key](name[len(key) :], params), wrapper_list[key](
                    "L1Loss"
                )

        return SimpleLoss(name, params), mae_func
    elif callable(name):
        return name, mae_func
    else:
        raise NotImplementedError(f"{name} Loss is not implemented")
