import logging

import torch

from nequip.nn import RescaleOutput, GraphModuleMixin, GradientOutput, PerSpeciesScaleShift,
from nequip.data import AtomicDataDict, AtomicDataset
from ._grads import ForceOutput

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


def PerSpecieRescale(
    model: GraphModuleMixin,
    config,
    dataset: AtomicDataset,
    initialize: bool,
):
    """Add global rescaling for energy(-based quantities).

    If ``initialize`` is false, doesn't compute statistics.
    """

    force_training = AtomicDataDict.FORCE_KEY in model.irreps_out

    # = Determine energy rescale type =
    global_scale = config.get(
        "global_rescale_scale",
        "dataset_force_rms" if force_training else "dataset_energy_std",)
    global_shift = config.get("global_rescale_shift", None)
    per_species_scale_shift = get_per_species(config, "enable", False)
    scales = pop_per_species(config, "scales", None)
    shifts = pop_per_species(config, "shifts", None)

    if global_shift is not None and per_species_scale_shift:
        raise ValueError("One can only enable either global shift or per_species shift")
    logging.info(f"Enable per species scale/shift: {per_species_scale_shift}")

    # = Determine what statistics need to be compute =
    keys = []
    for variable in [global_shift, global_scale, scales, shifts]:
        if isinstance(variable, str) and variable.startswith("dataset"):
            keys += [variable[len("dataset_") :]]
    keys = list(set(keys))
    if "force_rms" in keys and not force_training:
        raise ValueError(
            "Cannot have global_scale = 'dataset_force_rms' without force training"
        )

    # = Get statistics of training dataset =
    dataset_statistics = {}
    if "force_rms" in keys:
        ((rms,),) = trainer.dataset_train.statistics(
            fields=[AtomicDataDict.FORCE_KEY],
            modes=["rms"],
            stride=config.dataset_statistics_stride,
        )
        dataset_statistics["dataset_force_rms"] = rms
    if "energy_std" in keys or "energy_mean" in keys:
        ((mean, std),) = trainer.dataset_train.statistics(
            fields=[AtomicDataDict.TOTAL_ENERGY_KEY],

    # first peel off the gradient part
    if force_training:
        model = model.func
    
    # insert in per species shift
    model.insert_from_parameters(
        after="total_energy_sum",
        shared_params=config,
        name="per_species_scale_shift",
        builder=PerSpeciesScaleShift,
        params=dict(
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        ),
        prepend=True,)

    # wrap it back with gradient
    if force_training:
        model = ForceOutput(model)

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
        trainable_global_rescale_shift=False,
        trainable_global_rescale_scale=config.get(
            "trainable_global_rescale_scale", False
        ),
    )

def pop_per_species(config, key, default):
    return config.pop(
        f"PerSpeciesScaleShift_{key}",
        config.pop(f"per_species_scale_shift_{key}", default),
    )


def get_per_species(config, key, default):
    return config.get(
        f"PerSpeciesScaleShift_{key}",
        config.get(f"per_species_scale_shift_{key}", default),
    )


def set_value(variable, variable_name, value_dict):
    for key, value in value_dict.items():
        if variable == key:
            return value
    if (
        variable is None
        or isinstance(variable, float)
        or isinstance(variable, torch.Tensor)
    ):
        return variable
    raise ValueError(f"Invalid {variable_name} `{variable}`")