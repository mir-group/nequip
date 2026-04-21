# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch


def _normalize_weight_index_slices(weight_index_slices):
    normalized = []
    for entry in weight_index_slices:
        index_slice = getattr(entry, "slice_1D", None)
        shape_2d = getattr(entry, "shape_2D", None)
        if index_slice is None or shape_2d is None:
            index_slice, shape_2d = entry
        if isinstance(index_slice, slice):
            index_slice = (index_slice.start, index_slice.stop, index_slice.step)
        else:
            index_slice = tuple(index_slice)
        assert len(index_slice) == 3
        shape_2d = tuple(shape_2d)
        assert len(shape_2d) == 2
        normalized.append((index_slice, shape_2d))
    return normalized


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

            # use Muon only when reshape metadata is available
            weight_index_slices = getattr(module, "weight_index_slices", None)
            if weight_index_slices is None:
                adam_weights.append(param)
                continue

            # store plain tuples to keep optimizer state picklable
            index = len(muon_weights)
            e3nn_reshaping[index] = _normalize_weight_index_slices(weight_index_slices)
            muon_weights.append(param)
            continue

        adam_weights.append(param)

    param_groups = [
        dict(params=muon_weights, use_muon=True, e3nn_reshaping=e3nn_reshaping, **muon),
        dict(params=adam_weights, use_muon=False, **adam),
    ]

    return param_groups
