# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch


def MuonParamGroups(
    model: torch.nn.Module,
    muon: dict,
    adam: dict,
):
    """
    Build optimizer parameter groups, splitting parameters between a Muon-based optimizer
    and Adam (or Adam-like) optimizer.

    Assigned to Adam group:
      - Any parameter whose name does **not** contain the substring ``"layer"``.
      - Any parameter not matching the Muon-specific rules below.

    Assigned to Muon group:
      - Edge MLP weights: parameters whose name contains ``"edge_mlp"`` and that are
        2D tensors (i.e., matrix weights).
      - e3nn convolution linear weights: parameters whose name contains ``"conv.linear"``.

    For e3nn ``Linear`` layers, the returned Muon parameter group includes an
    ``e3nn_reshaping`` dictionary mapping the index of the parameter within the Muon
    group to the module's ``weight_index_slices``. This metadata will be used by the
     to reshape or operate on corresponding matrix weights.

    Args:
        model (torch.nn.Module): The model to optimize.
        muon (dict): Muon config parameters.
        adam (dict): Adam config parameters.

    """
    muon_weights = []
    adam_weights = []

    e3nn_reshaping = {}

    modules = dict(model.named_modules())

    for name, param in model.named_parameters():
        # Assumes all input and output layers are
        # not called layers.
        if "layer" not in name:
            adam_weights.append(param)
            continue

        # First, all edge_mlps should be muon
        if "edge_mlp" in name and param.ndim == 2:
            muon_weights.append(param)
            continue

        if "conv.linear" in name:
            # e3nn conv layers.

            # Find the e3nn Linear module this represents
            module_name, _, _ = name.rpartition(".")
            module = modules[module_name]

            # Attribute from e3nn giving the slices/shapes
            # of the corresponding linear weight
            index = len(muon_weights)
            slices = module.weight_index_slices

            e3nn_reshaping[index] = slices

            muon_weights.append(param)
            continue

        adam_weights.append(param)

    param_groups = [
        dict(params=muon_weights, use_muon=True, e3nn_reshaping=e3nn_reshaping, **muon),
        dict(params=adam_weights, use_muon=False, **adam),
    ]

    return param_groups
