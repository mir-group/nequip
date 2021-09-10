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
        stats = set()
        if global_scale == "dataset_energy_std":
            stats.add((AtomicDataDict.TOTAL_ENERGY_KEY, "mean_std"))
        elif global_scale == "dataset_force_rms":
            stats.add((AtomicDataDict.FORCE_KEY, "rms"))
        elif (
            global_scale is None
            or isinstance(global_scale, float)
            or isinstance(global_scale, torch.Tensor)
        ):
            # valid values
            pass
        else:
            raise ValueError(f"Invalid global scale `{global_scale}`")

        if global_shift == "dataset_energy_mean":
            stats.add((AtomicDataDict.TOTAL_ENERGY_KEY, "mean_std"))
        elif (
            global_shift is None
            or isinstance(global_shift, float)
            or isinstance(global_shift, torch.Tensor)
        ):
            # valid values
            pass
        else:
            raise ValueError(f"Invalid global shift `{global_shift}`")

        # = Compute shifts and scales =
        stats = list(stats)
        computed_stats = dataset.statistics(
            fields=[e[0] for e in stats],
            modes=[e[1] for e in stats],
            stride=config.dataset_statistics_stride,
        )

        def _find_stat(field, mode):
            return next(
                computed_stats[i]
                for i, (this_field, this_mode) in enumerate(stats)
                if this_field == field and this_mode == mode
            )

        if global_scale == "dataset_energy_std":
            global_scale = _find_stat(AtomicDataDict.TOTAL_ENERGY_KEY, "mean_std")[1]
        elif global_scale == "dataset_force_rms":
            global_scale = _find_stat(AtomicDataDict.FORCE_KEY, "rms")[0]

        if global_shift == "dataset_energy_mean":
            global_shift = _find_stat(AtomicDataDict.TOTAL_ENERGY_KEY, "mean_std")[0]

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
