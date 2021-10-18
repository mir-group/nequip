import logging
from typing import List, Optional

import torch

from nequip.nn import RescaleOutput, GraphModuleMixin, PerSpeciesScaleShift
from nequip.data import AtomicDataDict, AtomicDataset


RESCALE_THRESHOLD = 1e-6


def RescaleEnergyEtc(
    model: GraphModuleMixin,
    config,
    dataset: AtomicDataset,
    initialize: bool,
):
    """Add global rescaling for energy(-based quantities).

    If ``initialize`` is false, doesn't compute statistics.
    """

    module_prefix = "global_rescale"

    global_scale = config.get(
        f"{module_prefix}_scale",
        f"dataset_{AtomicDataDict.FORCE_KEY}_rms"
        if AtomicDataDict.FORCE_KEY in model.irreps_out
        else f"dataset_{AtomicDataDict.TOTAL_ENERGY_KEY}_std",
    )
    global_shift = config.get(f"{module_prefix}_shift", None)

    if global_shift is not None:
        logging.warning(
            f"!!!! Careful global_shift is set to {global_shift}."
            f"The energy model will no longer be extensive"
        )

    # = Get statistics of training dataset =
    if initialize:
        str_names = []
        for value in [global_scale, global_shift]:
            if isinstance(value, str):
                str_names += [value]
            elif (
                value is None
                or isinstance(value, float)
                or isinstance(value, torch.Tensor)
            ):
                # valid values
                pass
            else:
                raise ValueError(f"Invalid global scale `{value}`")

        # = Compute shifts and scales =
        computed_stats = _compute_stats(
            str_names=str_names,
            dataset=dataset,
            stride=config.dataset_statistics_stride,
        )

        if isinstance(global_scale, str):
            global_scale = computed_stats[str_names.index(global_scale)]
        if isinstance(global_shift, str):
            global_shift = computed_stats[str_names.index(global_shift)]

        if isinstance(global_scale, float) and global_scale < RESCALE_THRESHOLD:
            raise ValueError(
                f"Global energy scaling was very low: {global_scale}. If dataset values were used, does the dataset contain insufficient variation? Maybe try disabling global scaling with global_scale=None."
            )

        logging.debug(
            f"Initially outputs are globally scaled by: {global_scale}, total_energy are globally shifted by {global_shift}."
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
        shift_trainable=config.get(f"{module_prefix}_shift_trainable", False),
        scale_trainable=config.get(f"{module_prefix}_scale_trainable", False),
    )


def PerSpeciesRescale(
    model: GraphModuleMixin,
    config,
    dataset: AtomicDataset,
    initialize: bool,
):
    """Add global rescaling for energy(-based quantities).

    If ``initialize`` is false, doesn't compute statistics.
    """
    module_prefix = "per_species_rescale"

    # TO DO, how to make the default consistent with the global scale function?
    force_training = config.get(
        "force_training", "ForceOutput" in config["model_builders"]
    )

    # = Determine energy rescale type =
    scales = config.get(
        module_prefix + "_scales",
        f"dataset_{AtomicDataDict.FORCE_KEY}_rms"
        if force_training
        else f"dataset_per_atom_{AtomicDataDict.TOTAL_ENERGY_KEY}_std",
    )
    shifts = config.get(
        module_prefix + "_shifts",
        f"dataset_per_atom_{AtomicDataDict.TOTAL_ENERGY_KEY}_mean",
    )

    # = Determine what statistics need to be compute =
    if initialize:
        str_names = []
        for value in [scales, shifts]:
            if isinstance(value, str):
                str_names += [value]
            elif (
                value is None
                or isinstance(value, float)
                or isinstance(value, list)
                or isinstance(value, torch.Tensor)
            ):
                # valid values
                pass
            else:
                raise ValueError(f"Invalid value `{value}` of type {type(value)}")

        # = Compute shifts and scales =
        computed_stats = _compute_stats(
            str_names=str_names,
            dataset=dataset,
            stride=config.dataset_statistics_stride,
            kwargs=config.get(module_prefix + "_kwargs", {}),
        )

        if isinstance(scales, str):
            scales = computed_stats[str_names.index(scales)]
        elif isinstance(scales, list):
            scales = torch.as_tensor(scales)

        if isinstance(shifts, str):
            shifts = computed_stats[str_names.index(shifts)]
        elif isinstance(shifts, list):
            shifts = torch.as_tensor(shifts)

        if scales.min() < RESCALE_THRESHOLD:
            raise ValueError(
                f"Per species energy scaling was very low: {scales}. Maybe try setting {module_prefix}scales = 1."
            )

    else:

        # Put dummy values
        scales = None
        shifts = None

    # insert in per species shift
    model.insert_from_parameters(
        before="total_energy_sum",
        name=module_prefix,
        shared_params=config,
        builder=PerSpeciesScaleShift,
        params=dict(
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            shifts=shifts,
            scales=scales,
        ),
    )

    logging.debug(f"Atomic outputs are scaled by: {scales}, shifted by {shifts}.")

    # == Build the model ==
    return model


def _compute_stats(
    str_names: List[str], dataset, stride: int, kwargs: Optional[dict] = {}
):
    """return the values of statistics over dataset
    quantity name should be dataset_key_stat, where key can be any key
    that exists in the dataset, stat can be mean, std

    Args:

    str_names: list of strings that define the quantity to compute
    dataset: dataset object to run the stats over
    stride: # frames to skip for every one frame to include
    """

    # parse the list of string to field, mode
    # and record which quantity correspond to which computed_item
    stat_modes = []
    stat_fields = []
    stat_strs = []
    ids = []
    tuple_ids = []
    tuple_id_map = {"mean": 0, "std": 1, "rms": 0}
    input_kwargs = {}
    for name in str_names:

        # remove dataset prefix
        if name.startswith("dataset_"):
            name = name[len("dataset_") :]
        # identify per_species and per_atom modes
        prefix = ""
        if name.startswith("per_species_"):
            name = name[len("per_species_") :]
            prefix = "per_species_"
        elif name.startswith("per_atom_"):
            name = name[len("per_atom_") :]
            prefix = "per_atom_"

        stat = name.split("_")[-1]
        field = "_".join(name.split("_")[:-1])
        if stat in ["mean", "std"]:
            stat_mode = prefix + "mean_std"
            stat_str = field + prefix + "mean_std"
        elif stat in ["rms"]:
            stat_mode = prefix + "rms"
            stat_str = field + prefix + "rms"
        else:
            raise ValueError(f"Cannot handle {stat} type quantity")

        if stat_str in stat_strs:
            ids += [stat_strs.index(stat_str)]
        else:
            ids += [len(stat_strs)]
            stat_strs += [stat_str]
            stat_modes += [stat_mode]
            stat_fields += [field]
            if stat_mode.startswith("per_species_"):
                if field in kwargs:
                    input_kwargs[field + stat_mode] = kwargs[field]
        tuple_ids += [tuple_id_map[stat]]

    values = dataset.statistics(
        fields=stat_fields,
        modes=stat_modes,
        stride=stride,
        kwargs=input_kwargs,
    )
    return [values[idx][tuple_ids[i]] for i, idx in enumerate(ids)]
