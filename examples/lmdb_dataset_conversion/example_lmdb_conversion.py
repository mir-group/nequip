"""
Example of how to use NequIPLMDBDataset to convert an xyz file to LMDB format, using ase and some nequip ase utilities. This can be adapted to convert data from other formats, as long as there one writes code to convert data from the custom format to nequip's AtomicDataDict format.
"""

import torch
import ase
from nequip.data.dataset import NequIPLMDBDataset
from nequip.data.ase import from_ase
from nequip.utils import download_url

import os
from tqdm import tqdm


lmdb_file_path = "fcu.lmdb"
xyz_file_path = "fcu.xyz"


# make sure the data is saved in float64!
torch.set_default_dtype(torch.float64)

# === download example xyz file ===
# skip if one has their own xyz file to run this example
url = "https://archive.materialscloud.org/record/file?record_id=1302&filename=fcu.xyz"
_ = download_url(url, os.getcwd(), filename=xyz_file_path)

# === ase.Atoms -> AtomicDataDict ===
atoms_list = list(
    tqdm(
        ase.io.iread(filename=xyz_file_path, parallel=False),
        desc="Reading dataset with ASE...",
    )
)
atomic_data_dicts = (
    from_ase(atoms) for atoms in tqdm(atoms_list, desc="Saving to LMDB...")
)

# === convert to LMDB ===
NequIPLMDBDataset.save_from_iterator(
    file_path=lmdb_file_path,
    iterator=atomic_data_dicts,
    write_frequency=5000,  # increase this from default 1000 to speed up writing of very large datasets
)
