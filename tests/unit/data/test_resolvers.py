import pytest
from typing import Final, List
from omegaconf import OmegaConf

DATASETS: Final[List[str]] = ["MPTrj", "OMat"]
CUTOFFS: Final[List[float]] = [
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
]

REQUIRED_KEYS: Final[List[str]] = [
    "chemical_symbols",
    "per_atom_energy_mean",
    "forces_rms",
    "per_type_energy_shifts",
    "per_type_energy_scales",
    "num_neighbors_mean",
    "per_type_num_neighbours_mean",
]


@pytest.mark.parametrize("cutoff", CUTOFFS)
@pytest.mark.parametrize("dataset", DATASETS)
def test_big_dataset_stats_resolver(cutoff, dataset):
    """Tests `nequip.utils.resolvers.big_dataset_stats` resolver."""
    # Retrieve the resolved instance
    cfg = OmegaConf.create(
        {
            "dataset_stats": "${big_dataset_stats:" + dataset + "," + str(cutoff) + "}",
        }
    )

    dataset_stats = cfg.dataset_stats

    # The resolver should produce a dict-like OmegaConf node
    assert OmegaConf.is_config(dataset_stats)

    # Required keys
    for k in REQUIRED_KEYS:
        assert k in dataset_stats, f"missing key: {k}"

    # chemical_symbols: List[str]
    chem = dataset_stats.chemical_symbols

    assert OmegaConf.is_list(chem)
    assert all(isinstance(s, str) for s in chem)

    # Scalars
    assert isinstance(dataset_stats.per_atom_energy_mean, float)
    assert isinstance(dataset_stats.forces_rms, float)
    assert isinstance(dataset_stats.num_neighbors_mean, float)

    # per_type_energy_shifts: DictConfig[str, float]
    shifts = dataset_stats.per_type_energy_shifts
    assert OmegaConf.is_dict(shifts)
    assert all(isinstance(k, str) for k in shifts)
    assert all(isinstance(v, float) for v in shifts.values())

    assert set(shifts.keys()) == set(chem), (
        "All chemical_symbols must be represented in per_type_energy_shifts"
    )

    # per_type_energy_scales: DictConfig[str, float]
    scales = dataset_stats.per_type_energy_scales
    assert OmegaConf.is_dict(scales)
    assert all(isinstance(v, float) for v in scales.values())
    # This checks both that all keys are string and that all chemical symbols are represented
    assert set(scales.keys()) == set(chem), (
        "All chemical_symbols must be represented in per_type_energy_scales"
    )

    # per_type_num_neighbours_mean: DictConfig[str, float]
    pt_nn = dataset_stats.per_type_num_neighbours_mean
    assert OmegaConf.is_dict(pt_nn)
    assert all(isinstance(v, float) for v in pt_nn.values())
    assert set(pt_nn.keys()) == set(chem), (
        "All chemical_symbols must be represented in per_type_num_neighbours_mean"
    )

    # All per-type dicts must have the same length as chemical_symbols
    expected_len = len(chem)
    assert len(shifts) == expected_len
    assert len(scales) == expected_len
    assert len(pt_nn) == expected_len
