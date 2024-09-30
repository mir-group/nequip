import torch

from nequip.nn import PerTypeScaleShift as PerTypeScaleShiftModule
from nequip.nn import RescaleOutput as RescaleOutputModule
from nequip.nn import GraphModuleMixin
from nequip.data import AtomicDataDict
from nequip.utils import format_type_vals

from omegaconf import ListConfig
import warnings
from typing import Union

from nequip.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


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
        default_scale=None,
        default_scale_keys=AtomicDataDict.ALL_ENERGY_KEYS,
    )


def GlobalRescale(
    model: GraphModuleMixin,
    config,
    initialize: bool,
    module_prefix: str,
    default_scale: Union[str, float, list],
    default_scale_keys: list,
):
    """Add global rescaling for energy(-based quantities).

    If ``initialize`` is false, doesn't compute statistics.
    """
    global_scale = config.get(f"{module_prefix}_scale", default_scale)

    if global_scale is None:
        warnings.warn(
            f"Module `{module_prefix}` added but global_scale is `None`. Please check to ensure this is intended. To set global_scale, `{module_prefix}_global_scale` must be provided in the config."
        )

    # = Get statistics of training dataset =
    if initialize:
        if global_scale is not None and global_scale < RESCALE_THRESHOLD:
            raise ValueError(
                f"Global energy scaling was very low: {global_scale}. If dataset values were used, does the dataset contain insufficient variation? Maybe try disabling global scaling with global_scale=None."
            )

        logger.info(f"Outputs are globally scaled by: {global_scale}.")

    else:
        # Put dummy values
        if global_scale is not None:
            global_scale = 1.0

    error_string = "keys need to be a list"
    assert isinstance(default_scale_keys, list) or isinstance(
        default_scale_keys, ListConfig
    ), error_string

    # == Build the model ==
    return RescaleOutputModule(
        model=model,
        scale_keys=[k for k in default_scale_keys if k in model.irreps_out],
        scale_by=global_scale,
        default_dtype=config.get("default_dtype", None),
    )


def PerTypeEnergyScaleShift(
    model: GraphModuleMixin,
    config,
    initialize: bool,
):
    """Add per-atom rescaling and shifting for per-atom energies."""
    return _PerTypeScaleShift(
        scales_default=None,
        shifts_default=None,
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        module_prefix="per_type_energy_scale_shift",
        insert_before="total_energy_sum",
        model=model,
        config=config,
    )


def _PerTypeScaleShift(
    scales_default,
    shifts_default,
    field: str,
    out_field: str,
    module_prefix: str,
    insert_before: str,
    model: GraphModuleMixin,
    config,
):
    """Add per-atom rescaling and shifting for a field

    If ``initialize`` is false, doesn't compute statistics.
    """
    scales = config.get(module_prefix + "_scales", scales_default)
    shifts = config.get(module_prefix + "_shifts", shifts_default)

    if scales is None and shifts is None:
        warnings.warn(
            f"Module `{module_prefix}` added but both scales and shifts are `None`. Please check to ensure this is intended. To set scales and/or shifts, `{module_prefix}_scales` and/or `{module_prefix}_shifts` must be provided in the config."
        )

    for value in [scales, shifts]:
        if not (
            value is None
            or any(
                [
                    isinstance(value, val_type)
                    for val_type in [float, list, torch.Tensor, ListConfig]
                ]
            )
        ):
            raise ValueError(f"Invalid value `{value}` of type {type(value)}")

    scales = 1.0 if scales is None else scales
    scales = torch.as_tensor(scales)
    shifts = 0.0 if shifts is None else shifts
    shifts = torch.as_tensor(shifts)

    # TODO kind of weird error to check for here
    if scales is not None and torch.min(scales) < RESCALE_THRESHOLD:
        raise ValueError(
            f"Per species scaling was very low: {scales}. Maybe try setting {module_prefix}_scales = 1."
        )

    scale_str = format_type_vals(scales, config["type_names"])
    shift_str = format_type_vals(shifts, config["type_names"])
    logger.info(
        "Atomic outputs are \n" f"scaled by : {scale_str}\n" f"shifted by: {shift_str}"
    )

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
        builder=PerTypeScaleShiftModule,
        params=params,
    )
    return model
