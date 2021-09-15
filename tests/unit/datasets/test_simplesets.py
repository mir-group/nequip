import pytest

from os.path import isdir
from shutil import rmtree

from nequip.utils.config import Config
from nequip.data import dataset_from_config


include_frames = [0, 1]


@pytest.mark.parametrize("name", ["aspirin"])
def test_simple(name, temp_data, BENCHMARK_ROOT):

    config = Config(
        dict(
            dataset=name,
            root=f"{temp_data}/{name}",
            extra_fixed_fields={"r_max": 3},
            include_frames=include_frames,
            chemical_symbol_to_type={"H": 0, "C": 1, "O": 2},
        )
    )

    if name == "aspirin":
        config.dataset_file_name = str(BENCHMARK_ROOT / "aspirin_ccsd-train.npz")

    a = dataset_from_config(config)
    print(a.data)
    print(a.fixed_fields)
    assert isdir(config.root)
    assert isdir(a.processed_dir)
    assert len(a.data.edge_index) == len(include_frames)
    rmtree(a.processed_dir)
