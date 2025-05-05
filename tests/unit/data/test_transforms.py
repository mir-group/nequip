import torch
from nequip.data import AtomicDataDict, from_dict
from nequip.data.transforms import VirialToStressTransform, StressSignFlipTransform


def test_VirialToStressTransform():
    # create data
    num_frames = 3
    num_atoms = 17
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(num_atoms, 3),
        AtomicDataDict.CELL_KEY: torch.randn(num_frames, 3, 3),
        AtomicDataDict.PBC_KEY: torch.full((num_frames, 3), True),
        AtomicDataDict.VIRIAL_KEY: torch.randn(num_frames, 3, 3),
        AtomicDataDict.NUM_NODES_KEY: torch.tensor([4, 6, 7]),
    }
    # not strictly needed, but useful to test that `from_dict` runs with the given input dict
    data = from_dict(data)
    transformed = VirialToStressTransform()(data)
    # implementation is trivial, good enough to test that it runs and has the correct shape
    assert AtomicDataDict.STRESS_KEY in transformed
    assert transformed[AtomicDataDict.STRESS_KEY].shape == (num_frames, 3, 3)


def test_StressSignFlipTransform():
    # create data
    num_frames = 3
    num_atoms = 17
    stress = torch.randn(num_frames, 3, 3)
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(num_atoms, 3),
        AtomicDataDict.NUM_NODES_KEY: torch.tensor([4, 6, 7]),
        AtomicDataDict.PBC_KEY: torch.full((num_frames, 3), True),
        AtomicDataDict.STRESS_KEY: stress.clone(),
    }
    data = from_dict(data)
    result = StressSignFlipTransform()(data)[AtomicDataDict.STRESS_KEY]
    assert result.shape == (num_frames, 3, 3)
    assert torch.allclose(result, -stress)
