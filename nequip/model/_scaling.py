import logging
from typing import List, Optional, Union

import torch

from nequip.nn import RescaleOutput, GraphModuleMixin, PerSpeciesScaleShift
from nequip.data import AtomicDataDict, AtomicDataset
from nequip.data.transforms import TypeMapper


RESCALE_THRESHOLD = 1e-6


def RescaleEnergyEtc(
    model: GraphModuleMixin, config, dataset: AtomicDataset, initialize: bool
):
    return GlobalRescale(
        model=model,
        config=config,
        dataset=dataset,
        initialize=initialize,
        module_prefix="global_rescale",
        default_scale=f"dataset_{AtomicDataDict.FORCE_KEY}_rms"
        if AtomicDataDict.FORCE_KEY in model.irreps_out
        else f"dataset_{AtomicDataDict.TOTAL_ENERGY_KEY}_std",
        default_shift=None,
        default_scale_keys=AtomicDataDict.ALL_ENERGY_KEYS,
        default_shift_keys=[AtomicDataDict.TOTAL_ENERGY_KEY],
        default_related_scale_keys=[AtomicDataDict.PER_ATOM_ENERGY_KEY],
        default_related_shift_keys=[],
    )


def GlobalRescale(
    model: GraphModuleMixin,
    config,
    dataset: AtomicDataset,
    initialize: bool,
    module_prefix: str,
    default_scale: Union[str, float, list],
    default_shift: Union[str, float, list],
    default_scale_keys: list,
    default_shift_keys: list,
    default_related_scale_keys: list,
    default_related_shift_keys: list,
):
    """Add global rescaling for energy(-based quantities).

    If ``initialize`` is false, doesn't compute statistics.
    """

    global_scale = config.get(f"{module_prefix}_scale", default_scale)
    global_shift = config.get(f"{module_prefix}_shift", default_shift)

    if global_shift is not None:
        logging.warning(
            f"!!!! Careful global_shift is set to {global_shift}."
            f"The model for {default_shift_keys} will no longer be size extensive"
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
            s = global_scale
            global_scale = computed_stats[str_names.index(global_scale)]
            logging.info(f"Replace string {s} to {global_scale}")
        if isinstance(global_shift, str):
            s = global_shift
            global_shift = computed_stats[str_names.index(global_shift)]
            logging.info(f"Replace string {s} to {global_shift}")

        if isinstance(global_scale, float) and global_scale < RESCALE_THRESHOLD:
            raise ValueError(
                f"Global energy scaling was very low: {global_scale}. If dataset values were used, does the dataset contain insufficient variation? Maybe try disabling global scaling with global_scale=None."
            )

        logging.info(
            f"Initially outputs are globally scaled by: {global_scale}, total_energy are globally shifted by {global_shift}."
        )

    else:
        # Put dummy values
        if global_shift is not None:
            global_shift = 0.0  # it has some kind of value
        if global_scale is not None:
            global_scale = 1.0  # same,

    error_string = "keys need to be a list"
    assert isinstance(default_scale_keys, list), error_string
    assert isinstance(default_shift_keys, list), error_string
    assert isinstance(default_related_scale_keys, list), error_string
    assert isinstance(default_related_shift_keys, list), error_string

    # == Build the model ==
    return RescaleOutput(
        model=model,
        scale_keys=[k for k in default_scale_keys if k in model.irreps_out],
        scale_by=global_scale,
        shift_keys=[k for k in default_shift_keys if k in model.irreps_out],
        shift_by=global_shift,
        related_scale_keys=default_related_scale_keys,
        related_shift_keys=default_related_shift_keys,
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

    # = Determine energy rescale type =
    scales = config.get(
        module_prefix + "_scales",
        f"dataset_{AtomicDataDict.FORCE_KEY}_rms"
        # if `train_on_keys` isn't provided, assume conservatively
        # that we aren't "training" on anything (i.e. take the
        # most general defaults)
        if AtomicDataDict.FORCE_KEY in config.get("train_on_keys", [])
        else f"dataset_per_atom_{AtomicDataDict.TOTAL_ENERGY_KEY}_std",
    )
    shifts = config.get(
        module_prefix + "_shifts",
        f"dataset_per_atom_{AtomicDataDict.TOTAL_ENERGY_KEY}_mean",
    )

    # Check for common double shift mistake with defaults
    if "RescaleEnergyEtc" in config.get("model_builders", []):
        # if the defaults are enabled, then we will get bad double shift
        # THIS CHECK IS ONLY GOOD ENOUGH FOR EMITTING WARNINGS
        has_global_shift = config.get("global_rescale_shift", None) is not None
        if has_global_shift:
            if shifts is not None:
                # using default of per_atom shift
                raise RuntimeError(
                    "A global_rescale_shift was provided, but the default per-atom energy shift was not disabled."
                )
        del has_global_shift

    # = Determine what statistics need to be compute =\
    arguments_in_dataset_units = None
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

        if len(str_names) == 2:
            # Both computed from dataset
            arguments_in_dataset_units = True
        elif len(str_names) == 1:
            if None in [scales, shifts]:
                # if the one that isnt str is null, it's just disabled
                # that has no units
                # so it's ok to have just one and to be in dataset units
                arguments_in_dataset_units = True
            else:
                assert config[
                    module_prefix + "_arguments_in_dataset_units"
                ], "Requested to set either the shifts or scales of the per_species_rescale using dataset values, but chose to provide the other in non-dataset units. Please give the explictly specified shifts/scales in dataset units and set per_species_rescale_arguments_in_dataset_units"

        # = Compute shifts and scales =
        computed_stats = _compute_stats(
            str_names=str_names,
            dataset=dataset,
            stride=config.dataset_statistics_stride,
            kwargs=config.get(module_prefix + "_kwargs", {}),
        )

        if isinstance(scales, str):
            s = scales
            scales = computed_stats[str_names.index(scales)].squeeze(-1)  # energy is 1D
            logging.info(f"Replace string {s} to {scales}")
        elif isinstance(scales, (list, float)):
            scales = torch.as_tensor(scales)

        if isinstance(shifts, str):
            s = shifts
            shifts = computed_stats[str_names.index(shifts)].squeeze(-1)  # energy is 1D
            logging.info(f"Replace string {s} to {shifts}")
        elif isinstance(shifts, (list, float)):
            shifts = torch.as_tensor(shifts)

        if scales is not None and torch.min(scales) < RESCALE_THRESHOLD:
            raise ValueError(
                f"Per species energy scaling was very low: {scales}. Maybe try setting {module_prefix}_scales = 1."
            )

    else:
        # Put dummy values
        # the real ones will be loaded from the state dict later
        # note that the state dict includes buffers,
        # so this is fine regardless of whether its trainable.
        scales = 1.0 if scales is not None else None
        shifts = 0.0 if shifts is not None else None
        # values correctly scaled according to where the come from
        # will be brought from the state dict later,
        # so what you set this to doesnt matter:
        arguments_in_dataset_units = False

    # insert in per species shift
    params = dict(
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        shifts=shifts,
        scales=scales,
    )

    params["arguments_in_dataset_units"] = arguments_in_dataset_units
    model.insert_from_parameters(
        before="total_energy_sum",
        name=module_prefix,
        shared_params=config,
        builder=PerSpeciesScaleShift,
        params=params,
    )

    logging.info(
        f"Atomic outputs are scaled by: {TypeMapper.format(scales, config.type_names)}, shifted by {TypeMapper.format(shifts, config.type_names)}."
    )

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
