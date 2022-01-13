from nequip.nn import GraphModuleMixin, GradientOutput
from nequip.nn import PartialForceOutput as PartialForceOutputModule
from nequip.data import AtomicDataDict


def ForceOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces to a model that predicts energy.

    Args:
        energy_model: the model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``GradientOutput`` wrapping ``energy_model``.
    """
    if AtomicDataDict.FORCE_KEY in model.irreps_out:
        raise ValueError("This model already has force outputs.")
    return GradientOutput(
        func=model,
        of=AtomicDataDict.TOTAL_ENERGY_KEY,
        wrt=AtomicDataDict.POSITIONS_KEY,
        out_field=AtomicDataDict.FORCE_KEY,
        sign=-1,  # force is the negative gradient
    )


def PartialForceOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces and partial forces to a model that predicts energy.

    Args:
        energy_model: the model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``GradientOutput`` wrapping ``energy_model``.
    """
    if (
        AtomicDataDict.FORCE_KEY in model.irreps_out
        or AtomicDataDict.PARTIAL_FORCE_KEY in model.irreps_out
    ):
        raise ValueError("This model already has force outputs.")
    return PartialForceOutputModule(func=model)
