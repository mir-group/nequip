from nequip.nn import GraphModuleMixin, GradientOutput
from nequip.nn import PartialForceOutput as PartialForceOutputModule
from nequip.nn import StressOutput as StressOutputModule
from nequip.nn import ParaStressOutput as ParaStressOutputModule
from nequip.data import AtomicDataDict


def ForceOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces to a model that predicts energy.

    Args:
        model: the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``GradientOutput`` wrapping ``model``.
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
        model: the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``GradientOutput`` wrapping ``model``.
    """
    if (
        AtomicDataDict.FORCE_KEY in model.irreps_out
        or AtomicDataDict.PARTIAL_FORCE_KEY in model.irreps_out
    ):
        raise ValueError("This model already has force outputs.")
    return PartialForceOutputModule(func=model)


def StressForceOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces and stresses to a model that predicts energy.

    Args:
        model: the model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``StressOutput`` wrapping ``model``.
    """
    if (
        AtomicDataDict.FORCE_KEY in model.irreps_out
        or AtomicDataDict.STRESS_KEY in model.irreps_out
    ):
        raise ValueError("This model already has force or stress outputs.")
    return StressOutputModule(func=model)


def ParaStressForceOutput(model: GraphModuleMixin) -> GradientOutput:
    r"""Add forces and stresses to a model that predicts energy.

    Args:
        model: the model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``StressOutput`` wrapping ``model``.
    """
    if (
        AtomicDataDict.FORCE_KEY in model.irreps_out
        or AtomicDataDict.STRESS_KEY in model.irreps_out
    ):
        raise ValueError("This model already has force or stress outputs.")
    return ParaStressOutputModule(func=model)
