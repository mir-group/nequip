# This file is a part of the `allegro` package. Please see LICENSE and README at the root for information on using it.
import torch


def MuonParamGroups(
    model: torch.nn.Module,
    muon: dict,
    adam: dict,
):
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
        dict(params=adam_weights, use_muon=False, e3nn_reshaping=None, **adam),
    ]

    return param_groups
