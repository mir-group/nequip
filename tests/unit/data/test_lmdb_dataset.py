import lmdb
import torch
import pickle
import pytest
import numpy as np
from nequip.data.dataset import NequIPLMDBDataset


def make_atomic_data(i: int) -> dict:
    # Minimal AtomicDataDict with only 'pos' field
    return {"pos": torch.tensor([[float(i), float(i), float(i)]], dtype=torch.float)}


def test_save_load_and_metadata(tmp_path):
    # Number of entries
    n = 10
    # Create an iterator of AtomicDataDicts
    iterator = (make_atomic_data(i) for i in range(n))

    # Define LMDB file path
    lmdb_path = str(tmp_path / "test_data.lmdb")

    # Save the iterator to LMDB (with metadata support)
    NequIPLMDBDataset.save_from_iterator(
        file_path=lmdb_path,
        iterator=iterator,
        map_size=10_000_000,  # 10 MB
        write_frequency=2,  # commit every 2 items
    )

    # Load the dataset
    ds = NequIPLMDBDataset(file_path=lmdb_path)

    # Basic data checks
    assert len(ds) == n
    for i in range(n):
        data = ds[i]
        assert "pos" in data
        expected = torch.tensor([[float(i), float(i), float(i)]], dtype=torch.float)
        assert torch.equal(data["pos"], expected)

    # Metadata checks
    # num_frames should equal total number of entries
    assert ds.get_metadata("num_frames") == n

    # num_atoms_per_entry should be a list of length n, each entry = 1
    num_atoms_list = ds.get_metadata("num_atoms_per_entry")
    assert isinstance(num_atoms_list, np.ndarray)
    assert len(num_atoms_list) == n
    assert all(x == 1 for x in num_atoms_list)

    # Individual metadata element access
    assert ds.get_metadata("num_atoms_per_entry", 0) == 1
    assert ds.get_metadata("num_atoms_per_entry", [2, 4, 6]) == [1, 1, 1]

    # Non-existent metadata returns None
    assert ds.get_metadata("does_not_exist") is None
    assert ds.get_metadata("does_not_exist", 0) is None

    # Out-of-bounds indexing should raise IndexError
    with pytest.raises(IndexError):
        _ = ds[n]
    with pytest.raises(IndexError):
        ds.get_data_list([n])


# To ensure old LMDB files without metadata still work
def test_lmdb_without_metadata(tmp_path):
    # Define LMDB file path
    lmdb_path = str(tmp_path / "test_data.lmdb")

    # Number of entries
    n = 5

    # Manually create an LMDB with no "__metadata__" key
    db = lmdb.open(
        lmdb_path, map_size=1_000_000, subdir=False, meminit=False, writemap=True
    )
    txn = db.begin(write=True)
    for i in range(n):
        # reuse the same minimal AtomicDataDict
        data = make_atomic_data(i)
        txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
    txn.commit()
    db.sync()
    db.close()

    # Load with our dataset class
    ds = NequIPLMDBDataset(file_path=lmdb_path)

    # It should fall back to stat()['entries'] and ignore metadata
    assert len(ds) == n

    # Metadata queries must return None
    assert ds.get_metadata("num_frames") is None
    assert ds.get_metadata("num_atoms_per_entry") is None

    # Data still loads correctly
    for i in range(n):
        pos = ds[i]["pos"]
        expected = torch.tensor([[float(i), float(i), float(i)]])
        assert torch.equal(pos, expected)

    # Out‐of‐bounds still raises
    with pytest.raises(IndexError):
        _ = ds[n]
