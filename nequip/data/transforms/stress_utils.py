# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from nequip.data import AtomicDataDict


class VirialToStressTransform:
    r"""Converts virials to stress and adds the stress to the ``AtomicDataDict``.

    Specifically implements

    .. math::
        \tau_{ij} = - \frac{\sigma_{ij}}{\Omega}

    where :math:`\tau_{ij}` is a virial component, :math:`\sigma_{ij}` is a stress component, and :math:`\Omega` is the volume of the cell.
    """

    def __init__(self):
        pass

    def __call__(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # see discussion in https://github.com/libAtoms/QUIP/issues/227 about sign convention
        # they say the standard convention is virial = -stress x volume
        # we assume that the AtomicDataDict contains virials
        cell = data[AtomicDataDict.CELL_KEY]  # (num_frames, 3, 3)
        vol = torch.linalg.det(cell).abs()  # (num_frames,)
        virials = data[AtomicDataDict.VIRIAL_KEY]
        stress = virials.neg().div(vol.view(-1, 1, 1))  # (num_frames, 3, 3)
        data[AtomicDataDict.STRESS_KEY] = stress
        return data


class StressSignFlipTransform:
    r"""Flips the sign of stress in the ``AtomicDataDict``.

    In the NequIP convention, positive diagonal components of the stress tensor implies that the system is under tensile strain and wants to compress, while a negative value implies that the system is under compressive strain and wants to expand.
    This transform can be applied to datasets that follow the opposite sign convention, so that the necessary sign flip happens on-the-fly during training and users can avoid having to generate a copy of the dataset with NequIP stress sign conventions.
    """

    def __init__(self):
        pass

    def __call__(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # see discussion in https://github.com/libAtoms/QUIP/issues/227 about sign convention
        data[AtomicDataDict.STRESS_KEY] = data[AtomicDataDict.STRESS_KEY].neg()
        return data
