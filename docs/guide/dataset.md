# Datasets

## Preparing your data

`nequip` supports a number of input formats, but for the vast majority of cases we recommend preparing/converting your data to the extXYZ format. extXYZ supports arbitrary fields, variable numbers of atoms, and can be easily coverted from other formats using `ase.io.write`:
```python
import ase.io

TODO
```

```{warning}
Some ASE parsers, in particular those for data from DFT codes, silently convert units. Make sure you know what units your XYZ output is in! See [Units](./conventions.md#units).
```

```{tip}
Data in `numpy`'s NPZ format (see `NpzDataset`) and in a specific HDF5 format (see `HDF5Dataset`) are supported, but both currently lack features supported by the extXYZ format.
```

## Specifying your dataset

`nequip` builds dataset from a set of configuration options, typically provided to `nequip-train` or `nequip-evaluate` in a YAML file.

If the training and validation datasets are from different raw files, the arguments for each dataset
can be set independently with the `dataset` and `validation_dataset` prefixes, respectively.
For example, `dataset_file_name` is used for training data and `validation_dataset_file_name` is for validation data.

`nequip` accepts all format that can be parsed by `ase.io.read` function, but we recommend `extxyz`.
For example, given an atomic data stored in `H2.extxyz` that looks like below:
```xyz
2
Properties=species:S:1:pos:R:3 energy=-10 pbc="F F F"
H       0.00000000       0.00000000       0.00000000
H       0.00000000       0.00000000       1.02000000
```
The YAML input file should be
```yaml
dataset: ase
dataset_file_name: H2.extxyz
ase_args:
  format: extxyz
chemical_symbols:
  - H
```

## Using the dataset

During the initialization of an `AtomicDataset` object, `nequip` reads the atomic structures from the specified source, pre-computes the neighbor list, and stores the results in ``AtomicData`` objects.
The computed results are then cached to folder at the path `root/processed_hashkey` folder, where `root` comes from the configuration and `processed_hashkey` depends on all the metadata of the dataset, such as the source file name, the cutoff radius, floating point precision, and so on.
As a result, when multiple training/evaluation runs use the same dataset, the neighbor list will only be computed once. The later runs will directly load the AtomicDataset object from the cache file to save computation time.

```{warning}
If you update your source raw data file but keep using the same filename, thus not changing any of the metadata that names the cache folder, `nequip` cannot automatically update the cached data and will not reflect your later changes to the source file.
```

```{tip}
When using extXYZ input files, `nequip` parallelizes the computation of the neighbor list. While `nequip` will make a best effort to get a reasonable number of cores from your OS or job scheduler, you can override this with the `NEQUIP_NUM_TASKS` environment variable.
```

```{tip}
Pre-processing the dataset and computing the neighborlists can be quite memory intensive for large datasets, can be sped up with more cores, and does not use the GPU. You can force a pre-processing of your dataset on a CPU node by running `nequip-benchmark my-train-config.yaml` in a separate job, and then reuse the cached dataset in your later `nequip-train my-train-config.yaml` job, which can often then be run with a lower CPU core and host memory limit.
```

## Custom fields
`nequip` can load custom / nonstandard fields from the dataset, but they have to be requested specifically. Given this `H2.extxyz`:
```xyz
2
Properties=species:S:1:pos:R:3 energy=-10 user_label=2.0 pbc="F F F"
H       0.00000000       0.00000000       0.00000000
H       0.00000000       0.00000000       1.02000000
```
The YAML input file could be
```yaml
dataset: ase
dataset_file_name: H2.extxyz
ase_args:
  format: extxyz
include_keys:
  - user_label
key_mapping:
  user_label: label0
chemical_symbols:
  - H
```
`key_mapping` optionally maps the name of a field in the input data into a field name in the processed dataset.

Custom fields must be registered as being per-atom, per-edge, or per-frame:
```yaml
graph_fields: ["label0"]  
```
They can also be registered as being integer type, instead of floating point:
```yaml
long_fields: ["something_else"]
```
