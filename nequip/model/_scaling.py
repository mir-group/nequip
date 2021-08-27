import logging

import torch

from nequip.nn import RescaleOutput, GraphModuleMixin
from nequip.data import AtomicDataDict, AtomicDataset


def RescaleEnergyEtc(
    model: GraphModuleMixin,
    config,
    dataset: AtomicDataset,
    initialize: bool,
):
    """Add global rescaling for energy(-based quantities).

    If ``initialize`` is false, doesn't compute statistics.
    """
    global_scale = config.get(
        "global_rescale_scale",
        "dataset_force_rms"
        if AtomicDataDict.FORCE_KEY in model.irreps_out
        else "dataset_energy_std",
    )
    # TODO: change this default?
    global_shift = config.get("global_rescale_shift", "dataset_energy_mean")

    # = Get statistics of training dataset =
    if initialize:
        stats_fields = []
        stats_modes = []
        if global_scale == "dataset_energy_std":
            stats_fields = [AtomicDataDict.TOTAL_ENERGY_KEY]
            stats_modes = ["mean_std"]
        elif global_scale == "dataset_force_rms":
            stats_fields = [AtomicDataDict.TOTAL_ENERGY_KEY, AtomicDataDict.FORCE_KEY]
            stats_modes = ["mean_std", "rms"]
        elif (
            global_scale is None
            or isinstance(global_scale, float)
            or isinstance(global_scale, torch.Tensor)
        ):
            # valid values
            pass
        else:
            raise ValueError(f"Invalid global scale `{global_scale}`")

        stats = dataset.statistics(
            fields=stats_fields,
            modes=stats_modes,
            stride=config.dataset_statistics_stride,
        )
        ((energies_mean, energies_std),) = stats[:1]
        del stats_modes
        del stats_fields

        # = Determine shifts, scales =
        if global_scale == "dataset_energy_std":
            global_scale = energies_std
        elif global_scale == "dataset_force_rms":
            global_scale = stats[1][0]

        if global_shift == "dataset_energy_mean":
            global_shift = energies_mean
        elif (
            global_shift is None
            or isinstance(global_shift, float)
            or isinstance(global_shift, torch.Tensor)
        ):
            # valid values
            pass
        else:
            raise ValueError(f"Invalid global shift `{global_shift}`")

        RESCALE_THRESHOLD = 1e-6
        if isinstance(global_scale, float) and global_scale < RESCALE_THRESHOLD:
            raise ValueError(
                f"Global energy scaling was very low: {global_scale}. If dataset values were used, does the dataset contain insufficient variation? Maybe try disabling global scaling with global_scale=None."
            )

        logging.debug(
            f"Initially outputs are scaled by: {global_scale}, eneriges are shifted by {global_shift}."
        )
    else:
        # Put dummy values
        if global_shift is not None:
            global_shift = 0.0  # it has some kind of value
        if global_scale is not None:
            global_scale = 1.0  # same,

    # == Build the model ==
    return RescaleOutput(
        model=model,
        scale_keys=[
            k
            for k in (
                AtomicDataDict.TOTAL_ENERGY_KEY,
                AtomicDataDict.PER_ATOM_ENERGY_KEY,
                AtomicDataDict.FORCE_KEY,
            )
            if k in model.irreps_out
        ],
        scale_by=global_scale,
        shift_keys=[
            k for k in (AtomicDataDict.TOTAL_ENERGY_KEY,) if k in model.irreps_out
        ],
        shift_by=global_shift,
        trainable_global_rescale_shift=config.get(
            "trainable_global_rescale_shift", False
        ),
        trainable_global_rescale_scale=config.get(
            "trainable_global_rescale_scale", False
        ),
    )
