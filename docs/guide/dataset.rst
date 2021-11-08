.. _dataset_note:
   
How to prepare training dataset
=======================================

What NequIP do behind the scene
--------------

NequIP uses AtomicDataset class to store the atomic configurations. 
During the initialization of an AtomicDataset object, 
NequIP reads the atomic structure from the dataset, 
computes the neighbor list and other data structures needed for the GNN 
by converting raw data to `AtomicData` objects.

The computed results are then cached on harddisk `root/processed_hashkey` folder.
The hashing is based on all the metadata provided for the dataset, 
which includes all the arguments provided for the dataset, such as the file name, the cutoff radius, precision.
In the case where multiple training/evaluation use the same dataset,
the neighbor list will only be computed in the first NequIP run.
The later processes will load from the cached to save computation time.

Note: be careful to the cached file. If you update your raw data file but keep using the same filename,
NequIP will not automatically update the cached data.

Key concepts
--------------

fixed_fields
~~~~~~~~~~~~~~~~~~~~~~~~~
Fixed fields are the quantities that are shared among all the configurations in the dataset.
For example, if the dataset is a trajectory of an NVT MD simulation, the super cell size and the atomic species 
are indeed a constant matrix/vector through out the whole dataset.
In this case, in stead of repeating the same values for many times, 
we specify the cell and species as fixed fields and only provide them once.

yaml interface
~~~~~~~~~~~~~~~~~~~~~~~~~
`nequip-train` and `nequip-evaluate` automatically construct the AtomicDataset based on the yaml arguments.
Later sections offer a couple different examples.

If the training and validation datasets are from different raw files, the arguments for each set
can be defined with `dataset` prefix and `validation_dataset` prefix, respectively.

For example `dataset_file_name` is used for training data and `validation_dataset_file_name` is for validation data.

python interface
~~~~~~~~~~~~~~~~~~~~~~~~~
See `nequip.data.dataset.AtomicInMemoryDataset`.

Prepare dataset and specify in yaml config
--------------

ASE format
~~~~~~~~~~~~~~~~~~~~~~~~~

NequIP accept all format that can be parsed by `ase.io.read` function. 
We recommend `extxyz`.

Example: Given an atomic data stored in "H2.extxyz" that looks like below:

.. code:: extxyz
   2
   Properties=species:S:1:pos:R:3 energy=-10 user_label=2.0 pbc="F F F"
   H       0.00000000       0.00000000       0.00000000
   H       0.00000000       0.00000000       1.02000000

The yaml input should be

.. code:: yaml
   dataset: ase
   dataset_file_name: H2.extxyz
   ase_args:
   format: extxyz
   include_keys:
     - user_label
   key_mapping:
     user_label: label0
   chemical_symbol_to_type:
     H: 0

For other formats than `extxyz`, be careful to the ase parser, it may have different behavior than extxyz.
For example, the vasp parser store potential energy to `free_energy` instead of `energy`.
Because we optimize our code to the `extxyz` parser, we need some additional keys to help NequIP to understand the situtaion
Here's an example for vasp outcar. 

.. code:: yaml
   dataset: ase
   dataset_file_name: OUTCAR
   ase_args:
     format: vasp-out
   key_mapping:
     free_energy: total_energy
   chemical_symbol_to_type:
     H: 0

The way around is to use key mapping, please see more note below.

NPZ formate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your dataset constitute configurations that always have the same number of atoms, npz data format can be an option.

In the npz file, all the values should have the same row as the number of the configurations. 
For example, the force array of 36 atomic configurations of an N-atom system should have the shape of (36, N, 3);
their total_energy array should have the shape of (36).

Below is an example of the yaml specification.

.. code:: yaml
   dataset: npz
   dataset_file_name: example.npz
   include_keys:
     - user_label1
     - user_label2
   npz_fixed_field_keys:
     - cell
     - atomic_numbers
   key_mapping:
     position: pos
     force: forces
     energy: total_energy
     Z: atomic_numbers


Note on key mapping
~~~~~~~~~~~~~~~~~~~

NequIP has default key names for energy, force, cell (defined at nequip.data._keys)
Unlike in the ASE format where these information is automatically parsed,
in the npz data format, the correct key names have to be provided.
The common key names are: `total_energy`, `forces`, `atomic_numbers`, `pos`, `cell`, `pbc`.
the key_mapping can help to convert the user defined name (key) to NequIP default name (value).


Advanced options
----------------

skip frames during data processing
~~~~~~~~~~~~~~~~~~~~~~~~~
The `include_frame` argument can be specified in yaml to skip certain frames in the raw datafile.
The item has to be a list or a python iteratable object.

register graph, node, edge fields
~~~~~~~~~~~~~~~~~~~~~~~~~
Graph, node, edge fields are quantities that belong to 
the whole graph, each atom, each edge, respectively.
Example graph fields include cell, pbc, and total_energy.
Example node fields include pos, forces 

To help NequIP to properly assemble the batch data, graph quantity other than 
cell, pbc, total_energy should be registered.