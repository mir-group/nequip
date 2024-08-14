import logging
from typing import Union

import torch

from nequip.nn import RescaleOutput, GraphModuleMixin, PerSpeciesScaleShift
from nequip.data import AtomicDataDict


RESCALE_THRESHOLD = 1e-6


def RescaleEnergyEtc(
    model: GraphModuleMixin,
    config,
    initialize: bool,
):
    return GlobalRescale(
        model=model,
        config=config,
        initialize=initialize,
        module_prefix="global_rescale",
        default_scale=(
            f"dataset_{AtomicDataDict.FORCE_KEY}_rms"
            if AtomicDataDict.FORCE_KEY in model.irreps_out
            else f"dataset_{AtomicDataDict.TOTAL_ENERGY_KEY}_std"
        ),
        default_shift=None,
        default_scale_keys=AtomicDataDict.ALL_ENERGY_KEYS,
        default_shift_keys=[AtomicDataDict.TOTAL_ENERGY_KEY],
    )


def GlobalRescale(
    model: GraphModuleMixin,
    config,
    initialize: bool,
    module_prefix: str,
    default_scale: Union[str, float, list],
    default_shift: Union[str, float, list],
    default_scale_keys: list,
    default_shift_keys: list,
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
        for value in [global_scale, global_shift]:
            if not (
                value is None
                or isinstance(value, float)
                or isinstance(value, torch.Tensor)
            ):
                raise ValueError(f"Invalid global scale `{value}`")

        if global_scale is not None and global_scale < RESCALE_THRESHOLD:
            raise ValueError(
                f"Global energy scaling was very low: {global_scale}. If dataset values were used, does the dataset contain insufficient variation? Maybe try disabling global scaling with global_scale=None."
            )

        # TODO: who logs?
        # logging.info(
        #    f"Initially outputs are globally scaled by: {global_scale}, total_energy are globally shifted by {global_shift}."
        # )

    else:
        # Put dummy values
        if global_shift is not None:
            global_shift = 0.0  # it has some kind of value
        if global_scale is not None:
            global_scale = 1.0  # same,

    error_string = "keys need to be a list"
    assert isinstance(default_scale_keys, list), error_string
    assert isinstance(default_shift_keys, list), error_string

    # == Build the model ==
    return RescaleOutput(
        model=model,
        scale_keys=[k for k in default_scale_keys if k in model.irreps_out],
        scale_by=global_scale,
        shift_keys=[k for k in default_shift_keys if k in model.irreps_out],
        shift_by=global_shift,
        shift_trainable=config.get(f"{module_prefix}_shift_trainable", False),
        scale_trainable=config.get(f"{module_prefix}_scale_trainable", False),
        default_dtype=config.get("default_dtype", None),
    )


def PerSpeciesRescale(
    model: GraphModuleMixin,
    config,
    initialize: bool,
):
    """Add per-atom rescaling (and shifting) for per-atom energies."""
    module_prefix = "per_species_rescale"

    # Check for common double shift mistake with defaults
    if "RescaleEnergyEtc" in config.get("model_builders", []):
        # if the defaults are enabled, then we will get bad double shift
        # THIS CHECK IS ONLY GOOD ENOUGH FOR EMITTING WARNINGS
        has_global_shift = config.get("global_rescale_shift", None) is not None
        if has_global_shift:
            if config.get(module_prefix + "_shifts", True) is not None:
                # using default of per_atom shift
                raise RuntimeError(
                    "A global_rescale_shift was provided, but the default per-atom energy shift was not disabled."
                )
        del has_global_shift

    return _PerSpeciesRescale(
        scales_default=None,
        shifts_default=f"dataset_per_atom_{AtomicDataDict.TOTAL_ENERGY_KEY}_mean",
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        module_prefix=module_prefix,
        insert_before="total_energy_sum",
        model=model,
        config=config,
        initialize=initialize,
    )


def _PerSpeciesRescale(
    scales_default,
    shifts_default,
    field: str,
    out_field: str,
    module_prefix: str,
    insert_before: str,
    model: GraphModuleMixin,
    config,
    initialize: bool,
):
    """Add per-atom rescaling (and shifting) for a field

    If ``initialize`` is false, doesn't compute statistics.
    """
    scales = config.get(module_prefix + "_scales", scales_default)
    shifts = config.get(module_prefix + "_shifts", shifts_default)

    # = Determine what statistics need to be compute =
    assert config.get(
        module_prefix + "_arguments_in_dataset_units", True
    ), f"The PerSpeciesRescale builder is only compatible with {module_prefix + '_arguments_in_dataset_units'} set to True"

    if initialize:
        for value in [scales, shifts]:
            if not (
                value is None
                or isinstance(value, float)
                or isinstance(value, list)
                or isinstance(value, torch.Tensor)
            ):
                raise ValueError(f"Invalid value `{value}` of type {type(value)}")

        scales = torch.as_tensor(scales)
        shifts = torch.as_tensor(shifts)

        # TODO kind of weird error to check for here
        if scales is not None and torch.min(scales) < RESCALE_THRESHOLD:
            raise ValueError(
                f"Per species scaling was very low: {scales}. Maybe try setting {module_prefix}_scales = 1."
            )

        # TODO: who logs?
        # logging.info(
        #    f"Atomic outputs are scaled by: {TypeMapper.format(scales, config.type_names)}, shifted by {TypeMapper.format(shifts, config.type_names)}."
        # )

    else:
        # Put dummy values
        # the real ones will be loaded from the state dict later
        # note that the state dict includes buffers,
        # so this is fine regardless of whether its trainable.
        scales = 1.0 if scales is not None else None
        shifts = 0.0 if shifts is not None else None
        # values from the previously initialized model
        # will be brought in from the state dict later,
        # so these values (and rescaling them) doesn't matter

    # insert in per species shift
    params = dict(
        field=field,
        out_field=out_field,
        shifts=shifts,
        scales=scales,
        arguments_in_dataset_units=True,
    )
    model.insert_from_parameters(
        before=insert_before,
        name=module_prefix,
        shared_params=config,
        builder=PerSpeciesScaleShift,
        params=params,
    )

    # == Build the model ==
    return model
