import pytest
import torch

from nequip.nn.embedding.utils import (
    cutoff_partialdict_to_fulldict,
    cutoff_fulldict_to_tensor,
    cutoff_tensor_to_str,
    cutoff_str_to_fulldict,
    cutoff_partialdict_to_tensor,
)
from nequip.utils.global_dtype import _GLOBAL_DTYPE


def test_round_trip_two_types():
    """Test round-trip conversion for 2 atom types."""
    type_names = ["C", "O"]
    r_max = 6.0

    # start with partial dict
    partial_dict = {"C": 5.0, "O": {"C": 5.5, "O": 6.0}}

    # convert: partial_dict -> full_dict -> tensor -> str -> full_dict
    full_dict = cutoff_partialdict_to_fulldict(partial_dict, type_names, r_max)
    tensor = cutoff_fulldict_to_tensor(full_dict, type_names)
    metadata_str = cutoff_tensor_to_str(tensor)
    full_dict_reconstructed = cutoff_str_to_fulldict(metadata_str, type_names)

    # verify round-trip
    assert full_dict == full_dict_reconstructed

    # verify expected structure
    assert full_dict["C"]["C"] == 5.0
    assert full_dict["C"]["O"] == 5.0
    assert full_dict["O"]["C"] == 5.5
    assert full_dict["O"]["O"] == 6.0


def test_round_trip_three_types():
    """Test round-trip conversion for 3 atom types."""
    type_names = ["H", "C", "O"]
    r_max = 5.0

    # mixed float/dict specification
    partial_dict = {
        "H": 2.0,
        "C": {"H": 4.0, "C": 3.5, "O": 3.7},
        "O": 3.9,
    }

    # full round-trip
    full_dict = cutoff_partialdict_to_fulldict(partial_dict, type_names, r_max)
    tensor = cutoff_fulldict_to_tensor(full_dict, type_names)
    metadata_str = cutoff_tensor_to_str(tensor)
    full_dict_reconstructed = cutoff_str_to_fulldict(metadata_str, type_names)

    assert full_dict == full_dict_reconstructed


def test_round_trip_with_defaults():
    """Test that missing entries are filled with r_max."""
    type_names = ["H", "C", "O"]
    r_max = 5.0

    # partial spec with missing entries
    partial_dict = {"H": 2.0}

    full_dict = cutoff_partialdict_to_fulldict(partial_dict, type_names, r_max)

    # verify H entries are 2.0
    for target in type_names:
        assert full_dict["H"][target] == 2.0

    # verify C and O entries default to r_max
    for source in ["C", "O"]:
        for target in type_names:
            assert full_dict[source][target] == r_max


def test_tensor_format():
    """Test that tensor has correct shape and ordering."""
    type_names = ["C", "O"]
    full_dict = {
        "C": {"C": 5.0, "O": 4.5},
        "O": {"C": 4.5, "O": 6.0},
    }

    tensor = cutoff_fulldict_to_tensor(full_dict, type_names)

    # verify shape and dtype
    assert tensor.shape == (2, 2)
    assert tensor.dtype == _GLOBAL_DTYPE

    # verify row-major ordering
    assert tensor[0, 0].item() == 5.0  # C-C
    assert tensor[0, 1].item() == 4.5  # C-O
    assert tensor[1, 0].item() == 4.5  # O-C
    assert tensor[1, 1].item() == 6.0  # O-O


def test_str_format():
    """Test metadata string format."""
    tensor = torch.tensor([[5.0, 4.5], [4.5, 6.0]], dtype=_GLOBAL_DTYPE)
    metadata_str = cutoff_tensor_to_str(tensor)

    # should be space-separated row-major flattened
    assert metadata_str == "5.0 4.5 4.5 6.0"


def test_str_format_scalar():
    """Test that scalar tensors work."""
    scalar = torch.tensor(5.0, dtype=_GLOBAL_DTYPE)
    metadata_str = cutoff_tensor_to_str(scalar)
    assert metadata_str == "5.0"


def test_empty_str_returns_none():
    """Test that empty or None strings return None."""
    assert cutoff_str_to_fulldict("", ["C", "O"]) is None
    assert cutoff_str_to_fulldict(None, ["C", "O"]) is None


def test_wrong_number_of_values():
    """Test that wrong number of values raises assertion."""
    type_names = ["C", "O"]
    cutoff_str = "5.0 4.5 6.0"  # only 3 values, expected 4

    with pytest.raises(AssertionError, match="Expected 4 cutoff values"):
        cutoff_str_to_fulldict(cutoff_str, type_names)


def test_single_type():
    """Test conversion with single atom type."""
    type_names = ["O"]
    r_max = 4.0

    # test with float
    partial_dict = {"O": 2.0}
    full_dict = cutoff_partialdict_to_fulldict(partial_dict, type_names, r_max)
    assert full_dict == {"O": {"O": 2.0}}

    # test with dict
    partial_dict = {"O": {"O": 2.0}}
    full_dict = cutoff_partialdict_to_fulldict(partial_dict, type_names, r_max)
    assert full_dict == {"O": {"O": 2.0}}


def test_integer_uniform_cutoff():
    """Test that an integer uniform cutoff is treated like a float."""
    type_names = ["H", "C"]
    r_max = 4.0

    # YAML parses `H: 2` as an int
    partial_dict = {"H": 2}
    full_dict = cutoff_partialdict_to_fulldict(partial_dict, type_names, r_max)
    assert full_dict["H"] == {"H": 2.0, "C": 2.0}


def test_partial_missing_targets():
    """Test that missing target types default to r_max."""
    type_names = ["H", "C", "O"]
    r_max = 4.0

    partial_dict = {
        "H": {"C": 2.0},  # missing H->H and H->O
        "C": {"H": 3.0, "C": 3.5, "O": 3.7},
        "O": 3.9,
    }

    full_dict = cutoff_partialdict_to_fulldict(partial_dict, type_names, r_max)

    # H->C specified, others default to r_max
    assert full_dict["H"]["H"] == r_max
    assert full_dict["H"]["C"] == 2.0
    assert full_dict["H"]["O"] == r_max

    # C fully specified
    assert full_dict["C"]["H"] == 3.0
    assert full_dict["C"]["C"] == 3.5
    assert full_dict["C"]["O"] == 3.7

    # O uniform
    assert full_dict["O"]["H"] == 3.9
    assert full_dict["O"]["C"] == 3.9
    assert full_dict["O"]["O"] == 3.9


def test_extra_types_ignored():
    """Test that extra types in config are ignored."""
    type_names = ["H", "C", "O"]
    r_max = 4.0

    # N is in the config but not in type_names
    partial_dict = {
        "H": 2.0,
        "C": {"H": 4.0, "C": 3.5, "O": 3.7, "N": 3.7},
        "O": 3.9,
    }

    full_dict = cutoff_partialdict_to_fulldict(partial_dict, type_names, r_max)

    # should only have H, C, O keys
    assert set(full_dict.keys()) == {"H", "C", "O"}
    for source in full_dict:
        assert set(full_dict[source].keys()) == {"H", "C", "O"}


def test_cutoff_exceeds_rmax_fails():
    """Test that cutoffs exceeding r_max raise assertion error."""
    type_names = ["H", "C"]
    r_max = 4.0

    # cutoff for C-C exceeds r_max
    partial_dict = {"H": 2.0, "C": {"H": 3.0, "C": 5.0}}

    with pytest.raises(AssertionError):
        cutoff_partialdict_to_tensor(partial_dict, type_names, r_max)
