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
