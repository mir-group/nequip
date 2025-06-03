# Data Configuration

## Data Processing Flow

The data processing in NequIP follows this pipeline from raw files to model-ready data:

```{figure} ./ml-data-flow-chart.svg
:width: 400px
:align: center
```

This entire pipeline is coordinated and managed by a {class}`~nequip.data.datamodule.NequIPDataModule` object, which also:
- Manages train/val/test dataset splits  
- Computes dataset statistics

**Key Components:**

- **{class}`~nequip.data.datamodule.NequIPDataModule`**: Orchestrates everything
- **Dataset**: Reads raw data files and applies transforms to individual structures
- **Transforms**: Process data sequentially (e.g. compute neighbor lists, map atom types)
- **DataLoader**: Batches transformed data for efficient training with parallel loading
- **Statistics Manager**: Computes statistics of processed data to initialize model parameters

## DataModules

The [data section](config.md/#data) of the NequIP config file specifies a {class}`~nequip.data.datamodule.NequIPDataModule`, which manages how training data is loaded and processed.
{class}`~nequip.data.datamodule.NequIPDataModule`s coordinate all aspects of data handling from loading to preprocessing.
For comprehensive configuration options, see [`nequip.data.datamodule`](../api/datamodule.rst).

### Common DataModules

{class}`~nequip.data.datamodule.ASEDataModule` is the most commonly used datamodule because it can read many file formats through [ASE](https://wiki.fysik.dtu.dk/ase/) (Atomic Simulation Environment), including popular formats such as the `.xyz` format.
The following is an example of splitting a single data file into separate training, validation, and testing sets.

```yaml
data:
  _target_: nequip.data.datamodule.ASEDataModule
  split_dataset:
    file_path: training_data.xyz
    train: 0.8
    val: 0.1
    test: 0.1
  # ... other arguments
```

### Specialized DataModules

For specific benchmark datasets, specialized datamodules provide auto-download capabilities and predefined configurations:

- {class}`~nequip.data.datamodule.MD22DataModule` - MD22 datasets
- {class}`~nequip.data.datamodule.rMD17DataModule` - Revised MD17 datasets
- {class}`~nequip.data.datamodule.sGDML_CCSD_DataModule` - sGDML datasets
- {class}`~nequip.data.datamodule.TM23DataModule` - TM23 dataset
- {class}`~nequip.data.datamodule.NequIP3BPADataModule` - 3BPA dataset

These specialized datamodules have unique APIs tailored to their specific datasets and often handle downloading and preprocessing automatically.

### Custom Data Configurations

For more complex or custom data setups, you can use the base {class}`~nequip.data.datamodule.NequIPDataModule` directly. This allows you to specify custom dataset configurations - datasets are the components that actually read data files and apply transforms to individual structures. See [`nequip.data.dataset`](../api/dataset.rst) for available dataset classes.

The existing [specialized datamodules](#specialized-datamodules) are essentially convenience wrappers that simplify configuring the base {class}`~nequip.data.datamodule.NequIPDataModule` with specific datasets and common settings.

### DataModule Arguments

Key arguments that datamodules take include transforms (see [Data Transforms](#data-transforms)), dataloaders (see [DataLoaders](#dataloaders)), and dataset statistics managers (see [Dataset Statistics](#dataset-statistics)).

## Data Transforms

Transforms process raw data into a format suitable for model training. They are specified in datamodule configurations (see [DataModules](#datamodules) and [`nequip.data.datamodule`](../api/datamodule.rst)) which pass them as arguments to datasets (see [`nequip.data.dataset`](../api/dataset.rst)) where they are applied sequentially to each data point.
Two transforms are essential for most use cases:

- **{class}`~nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper`** (usually required) maps atomic numbers to model type indices. This handles the distinction between chemical elements (C, H, O) and the integer type indices (0, 1, 2) that the model uses:
  ```yaml
  - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
    chemical_symbols: [C, H, O, Cu]  # Order determines type indices
  ```

- **{class}`~nequip.data.transforms.NeighborListTransform`** (always required) computes which atoms are neighbors of each atom within a cutoff distance. This is fundamental for graph-based neural networks:
  ```yaml
  - _target_: nequip.data.transforms.NeighborListTransform
    r_max: 5.0  # should be the same as model `r_max`
  ```

The `chemical_symbols` list defines the mapping from atomic numbers to type indices, and `type_names` should be consistent across data, model, and statistics configurations.

Here's an example with both transforms:

```yaml
transforms:
  - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
    chemical_symbols: [C, H, O, Cu]
  - _target_: nequip.data.transforms.NeighborListTransform
    r_max: 5.0
```

```{warning}
**Transform Order May Matter**: The order of transforms can be important for some configurations. For example, when using per-edge-type cutoffs in {class}`~nequip.data.transforms.NeighborListTransform`, the {class}`~nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper` must come before {class}`~nequip.data.transforms.NeighborListTransform` because the neighborlist transform needs atom type information to apply different cutoffs for different element pairs.
```

Additional transforms are available for specific use cases. For stress-related data, you may need {class}`~nequip.data.transforms.VirialToStressTransform` (converts virial to stress tensors) or {class}`~nequip.data.transforms.StressSignFlipTransform` (handles different stress sign conventions). For a complete list of available transforms, see the [transforms API documentation](../api/data_transforms.rst).

## DataLoaders

DataLoaders handle batching and parallel data loading using PyTorch's {class}`torch.utils.data.DataLoader`.
They are specified in datamodule configurations (see [DataModules](#datamodules) and [`nequip.data.datamodule`](../api/datamodule.rst)) which use them to wrap datasets for efficient training:

```yaml
train_dataloader:
  _target_: torch.utils.data.DataLoader
  batch_size: 5        # an important training hyperparameter to tune
  num_workers: 5       # parallel workers for data loading
  shuffle: true        # often useful to shuffle training data
```

```{tip}
Training batch size affects learning dynamics and is an important hyperparameter to tune. However, validation and test batch sizes have no effect on training and should generally be set as large as possible without causing out-of-memory errors to speed up evaluation.
```

## Dataset Statistics

Dataset statistics provide both rough knowledge of your dataset (e.g., average energy per atom, force magnitudes) and are crucial for initializing data-derived model hyperparameters.
They are computed by specifying a dataset statistics manager as an argument to datamodules (see [DataModules](#datamodules) and [`nequip.data.datamodule`](../api/datamodule.rst)).
The {class}`~nequip.data.CommonDataStatisticsManager` automatically computes essential statistics:

```yaml
stats_manager:
  _target_: nequip.data.CommonDataStatisticsManager
  type_names: [C, H, O, Cu]
  dataloader_kwargs:
    batch_size: 10  # Can be larger than training batch size to speed up computation
```

You can use a larger `batch_size` in `dataloader_kwargs` than your training batch size to compute statistics faster without memory issues.
Statistics are computed once during data setup, not during training.

For advanced use cases, you should use the base {class}`~nequip.data.DataStatisticsManager` directly for more flexible configuration.
See the [dataset statistics API documentation](../api/data_stats.rst) for configuration options.

For guidance on using computed statistics to initialize model parameters, see [Training data statistics as hyperparameters](model.md/#training-data-statistics-as-hyperparameters).