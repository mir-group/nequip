# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Most recent change on the top.

## Unreleased


## [0.13.0]

### Added
- LAMMPS ML-IAP integration
- `LinearLossCoefficientScheduler` callback
- `TF32Scheduler` callback

### Changed
- `EMTTestDataset` now uses orthorhombic cells


## [0.12.1]

### Added
- `MaximumAbsoluteError` metric

### Changed
- Update `SoftAdapt` callback to weight loss coefficient updates by the chosen (initial) loss coefficients.

### Fixed
- `PerTypeScaleShift` model modifier shape bug
- `WandbWatch` callback typo in docstring
- Broken links to get `fcu.xyz` example dataset for tutorials/examples
- `nequip-package` will now always pick up OEQ file even if OEQ is not installed at package time


## [0.12.0]

### Added
- `SAM23DataModule`: Specialized datamodule for Samsung's SAMD23 dataset (HfO and SiN systems), with automatic download, extraction, and optional OOD test set support.

### Fixed
- Train-time compile compatibility of OpenEquivariance accelerated NequIP models

### Removed
- [Breaking] Python 3.9 support has been removed


## [0.11.1]

### Fixed
- Fixed per-edge-type cutoff metadata parsing when loading compiled models

### Changed
- Change to make train-time compile works with PyTorch 2.8.0 -- train-time compile won't work with PyTorch 2.8.0 and NequIP versions before v0.11.1


## [0.11.0]

### Added
- Energy-only training support: `NequIPGNNEnergyModel`, `EnergyOnlyDataStatisticsManager`, `EnergyOnlyLoss`, and `EnergyOnlyMetrics` for datasets without force labels
- Per-edge-type cutoff handling in ASE Calculator

### Removed
- [Future] Python 3.9 support will be removed in the coming releases. For now, users will be warned to upgrade if using NequIP with Python 3.9.

### Changed
- [Breaking] Hide `NequIPCalculator.from_checkpoint_model` and `NequIPCalculator.from_packaged_model` methods by making them private; users should only ever use `NequIPCalculator.from_compiled_model`


## [0.10.0]

### Changed
- Restructured user guide docs

### Added
- OpenEquivariance acceleration for NequIP GNN models
- `SortedNeighborListTransform` for sorted neighborlists with permutation indices to get a sorted transpose
- Per-edge-type cutoff-aware neighborlist transform


## [0.9.1]

### Changed
- Revamped docs


## [0.9.0]

### Changed
- [Breaking] `nequip-compile` CLI: `--input-path` and `--output-path` are now positional arguments instead of flags
- [Breaking] `nequip-package build` CLI: `--ckpt-path` and `--output-path` are now positional arguments instead of flags


## [0.8.0]

### Added
- MD22 datamodule
- `PerTypeScaleShift` model modifier to enable changing the per-type atomic energy scales and shifts of a pretrained model for fine-tuning

### Changed
- [Breaking] packaged model file metadata format: NOTE that packaged models before this version will no longer be compatible, and models must be repackaged from checkpoints
- [Breaking] `nequip-package` API: users must now specify `nequip-package build` to construct packaged model files or `nequip-package info` to inspect the metadata of packaged model files
- `per_type_energy_scales` and `per_type_energy_shifts` are expected to be in dict format; warnings will be thrown if they are provided as lists, and this will become errors in later major releases
- [Breaking] `InteractionBlock` of NequIP model refactored such that checkpoint files from previous versions will not work with this version

### Removed
- [Breaking] train-time TorchScript


## [0.7.1]

### Fixed
- Update outdated descriptions for `parity_plot.py`

### Added
- Increase docs coverage
- `StressSignFlipTransform` as a data transform to flip the sign of datasets that come with the opposite stress sign convention as used in the NequIP framework
- `int_div` and `int_mul` resolvers for integer arithmetic in config files, e.g. `half_width: ${int_div:${width},2}`

### Changed
- Renamed `examples` -> `misc` directory


## [0.7.0]
A major backwards-incompatible update with breaking changes throughout the code.


## [0.6.2] - 2025-3-22

### Fixed
 - Fixed early stopping bug
 - Fixed PyTorch version bug

## [0.6.1] - 2024-7-9
### Added
- add support for equivariance testing of arbitrary Cartesian tensor outputs
- [Breaking] use entry points for `nequip.extension`s (e.g. for field registration)
- alternate neighborlist support enabled with `NEQUIP_NL` environment variable, which can be set to `ase` (default), `matscipy` or `vesin`
- Allow `n_train` and `n_val` to be specified as percentages of datasets.
- Only attempt training restart if `trainer.pth` file present (prevents unnecessary crashes due to file-not-found errors in some cases)

### Changed
- [Breaking] `NEQUIP_MATSCIPY_NL` environment variable no longer supported

### Fixed
- Fixed `flake8` install location in `pre-commit-config.yaml`


## [0.6.0] - 2024-5-10
### Added
- add Tensorboard as logger option
- [Breaking] Refactor overall model logic into `GraphModel` top-level module
- [Breaking] Added `model_dtype`
- `BATCH_PTR_KEY` in `AtomicDataDict`
- `AtomicInMemoryDataset.rdf()` and `examples/rdf.py`
- `type_to_chemical_symbol`
- Pair potential terms
- `nequip-evaluate --output-fields-from-original-dataset`
- Error (or warn) on unused options in YAML that likely indicate typos
- `dataset_*_absmax` statistics option
- `HDF5Dataset` (#227)
- `include_file_as_baseline_config` for simple modifications of existing configs
- `nequip-deploy --using-dataset` to support data-dependent deployment steps
- Support for Gaussian Mixture Model uncertainty quantification (https://doi.org/10.1063/5.0136574)
- `start_of_epoch_callbacks`
- `nequip.train.callbacks.loss_schedule.SimpleLossSchedule` for changing the loss coefficients at specified epochs
- `nequip-deploy build --checkpoint` and `--override` to avoid many largely duplicated YAML files
- matscipy neighborlist support enabled with `NEQUIP_MATSCIPY_NL` environment variable

### Changed
- Always require explicit `seed`
- [Breaking] Set `dataset_seed` to `seed` if it is not explicitly provided
- Don't log as often by default
- [Breaking] Default nonlinearities are `silu` (`e`) and `tanh` (`o`)
- Will not reproduce previous versions' data shuffling order (for all practical purposes this does not matter, the `shuffle` option is unchanged)
- [Breaking] `default_dtype` defaults to `float64` (`model_dtype` default `float32`, `allow_tf32: true` by default--- see https://arxiv.org/abs/2304.10061)
- `nequip-benchmark` now only uses `--n-data` frames to build the model
- [Breaking] By default models now use `StressForceOutput`, not `ForceOutput`
- Added `edge_energy` to `ALL_ENERGY_KEYS` subjecting it to global rescale

### Fixed
- Work with `wandb>=0.13.8`
- Better error for standard deviation with too few data
- `load_model_state` GPU -> CPU
- No negative volumes in rare cases 

### Removed
- [Breaking] `fixed_fields` machinery (`npz_fixed_field_keys` is still supported, but through a more straightforward implementation)
- Default run name/WandB project name of `NequIP`, they must now always be provided explicitly
- [Breaking] Removed `_params` as an allowable subconfiguration suffix (i.e. instead of `optimizer_params` now only `optimizer_kwargs` is valid, not both)
- [Breaking] Removed `per_species_rescale_arguments_in_dataset_units`

## [0.5.6] - 2022-12-19
### Added
- sklearn dependency removed
- `nequip-benchmark` and `nequip-train` report number of weights and number of trainable weights
- `nequip-benchmark --no-compile` and `--verbose` and `--memory-summary`
- `nequip-benchmark --pdb` for debugging model (builder) errors
- More information in `nequip-deploy info`
- GPU OOM offloading mode

### Changed
- Minimum e3nn is now 0.4.4
- `--equivariance-test` now prints much more information, especially when there is a failure

### Fixed
- Git utilities when installed as ZIPed `.egg` (#264)

## [0.5.5] - 2022-06-20
### Added
- BETA! Support for stress in training and inference
- `EMTTestDataset` for quick synthetic fake PBC data
- multiprocessing for ASE dataset loading/processing
- `nequip-benchmark` times dataset loading, model creation, and compilation
- `validation_batch_size`
- support multiple metrics on same field with different `functional`s
- allow custom metrics names
- allow `e3nn==0.5.0`
- `--verbose` option to `nequip-deploy`
- print data statistics in `nequip-benchmark`
- `normalized_sum` reduction in `AtomwiseReduce`

### Changed
- abbreviate `node_features`->`h` in loss titles
- failure of permutation equivariance tests no longer short-circuts o3 equivariance tests
- `NequIPCalculator` now stores all relevant properties computed by the model regardless of requested `properties`, and does not try to access those not computed by the model, allowing models that only compute energy or forces but not both

### Fixed
- Equivariance testing correctly handles output cells
- Equivariance testing correctly handles one-node or one-edge data
- `report_init_validation` now runs on validation set instead of training set
- crash when unable to find `os.sched_getaffinity` on some systems
- don't incorrectly log per-species scales/shifts when loading model (such as for deployment)
- `nequip-benchmark` now picks data frames deterministically
- useful error message for `metrics_key: training_*` with `report_init_validation: True` (#213)

## [0.5.4] - 2022-04-12
### Added
- `NequIPCalculator` now handles per-atom energies
- Added `initial_model_state_strict` YAML option
- `load_model_state` builder
- fusion strategy support
- `cumulative_wall` for early stopping
- Deploy model from YAML file directly

### Changed
- Disallow PyTorch 1.9, which has some JIT bugs.
- `nequip-deploy build` now requires `--train-dir` option when specifying the training session
- Minimum Python version is now 3.7

### Fixed
- Better error in `Dataset.statistics` when field is missing
- `NequIPCalculator` now outputs energy as scalar rather than `(1, 1)` array
- `dataset: ase` now treats automatically adds `key_mapping` keys to `include_keys`, which is consistant with the npz dataset
- fixed reloading models with `per_species_rescale_scales/shifts` set to `null`/`None`
- graceful exit for `-n 0` in `nequip-benchmark`
- Strictly correct CSV headers for metrics (#198)

## [0.5.3] - 2022-02-23
### Added
- `nequip-evaluate --repeat` option
- Report number of weights to wandb

### Changed
- defaults and commments in example.yaml and full.yaml, in particular longer default training and correct comment for E:F-weighting
- better metrics config in example.yaml and full.yaml, in particular will total F-MAE/F-RMSE instead of mean over per-species
- default value for `report_init_validation` is now `True`
- `all_*_*` metrics rename to -> `psavg_*_*`
- `avg_num_neighbors` default `None` -> `auto`

### Fixed
- error if both per-species and global shift are used together


## [0.5.2] - 2022-02-04
### Added
- Model builders may now process only the configuration
- Allow irreps to optionally be specified through the simplified keys `l_max`, `parity`, and `num_features`
- `wandb.watch` via `wandb_watch` option
- Allow polynomial cutoff _p_ values besides 6.0
- `nequip-evaluate` now sets a default `r_max` taken from the model for the dataset config
- Support multiple rescale layers in trainer
- `AtomicData.to_ase` supports arbitrary fields
- `nequip-evaluate` can now output arbitrary fields to an XYZ file
- `nequip-evaluate` reports which frame in the original dataset was used as input for each output frame

### Changed
- `minimal.yaml`, `minimal_eng.yaml`, and `example.yaml` now use the simplified irreps options `l_max`, `parity`, and `num_features`
- Default value for `resnet` is now `False`

### Fixed
- Handle one of `per_species_shifts`/`scales` being `null` when the other is a dataset statistc
- `include_frames` now works with ASE datasets
- no training data labels in input_data
- Average number of neighbors no longer crashes sometimes when not all nodes have neighbors (small cutoffs)
- Handle field registrations correctly in `nequip-evaluate`

### Removed
- `compile_model`

## [0.5.1] - 2022-01-13
### Added
- `NequIPCalculator` can now be built via a `nequip_calculator()` function. This adds a minimal compatibility with [vibes](https://gitlab.com/vibes-developers/vibes/)
- Added `avg_num_neighbors: auto` option
- Asynchronous IO: during training, models are written asynchronously. Enable this with environment variable `NEQUIP_ASYNC_IO=true`.
- `dataset_seed` to separately control randomness used to select training data (and their order).
- The types may now be specified with a simpler `chemical_symbols` option
- Equivariance testing reports per-field errors
- `--equivariance-test n` tests equivariance on `n` frames from the training dataset

### Changed
- All fields now have consistant [N, dim] shaping
- Changed default `seed` and `dataset_seed` in example YAMLs
- Equivariance testing can only use training frames now

### Fixed
- Equivariance testing no longer unintentionally skips translation
- Correct cat dim for all registered per-graph fields
- `PerSpeciesScaleShift` now correctly outputs when scales, but not shifts, are enabled— previously it was broken and would only output updated values when both were enabled.
- `nequip-evaluate` outputs correct species to the `extxyz` file when a chemical symbol <-> type mapping exists for the test dataset

## [0.5.0] - 2021-11-24
### Changed
- Allow e3nn 0.4.*, which changes the default normalization of `TensorProduct`s; this change _should_ not affect typical NequIP networks
- Deployed are now frozen on load, rather than compile

### Fixed
- `load_deployed_model` respects global JIT settings

## [0.4.0] - not released
### Added
- Support for `e3nn`'s `soft_one_hot_linspace` as radial bases
- Support for parallel dataloader workers with `dataloader_num_workers`
- Optionally independently configure validation and training datasets
- Save dataset parameters along with processed data
- Gradient clipping
- Arbitrary atom type support
- Unified, modular model building and initialization architecture
- Added `nequip-benchmark` script for benchmarking and profiling models
- Add before option to SequentialGraphNetwork.insert
- Normalize total energy loss by the number of atoms via PerAtomLoss
- Model builder to initialize training from previous checkpoint
- Better error when instantiation fails
- Rename `npz_keys` to `include_keys`
- Allow user to register `graph_fields`, `node_fields`, and `edge_fields` via yaml
- Deployed models save the e3nn and torch versions they were created with

### Changed
- Update example.yaml to use wandb by default, to only use 100 epochs of training, to set a very large batch logging frequency and to change Validation_loss to validation_loss
- Name processed datasets based on a hash of their parameters to ensure only valid cached data is used
- Do not use TensorFloat32 by default on Ampere GPUs until we understand it better
- No atomic numbers in networks
- `dataset_energy_std`/`dataset_energy_mean` to `dataset_total_energy_*`
- `nequip.dynamics` -> `nequip.ase`
- update example.yaml and full.yaml with better defaults, new loss function, and switched to toluene-ccsd(t) as example 
data
- `use_sc` defaults to `True`
- `register_fields` is now in `nequip.data`
- Default total energy scaling is changed from global mode to per species mode.
- Renamed `trainable_global_rescale_scale` to `global_rescale_scale_trainble`
- Renamed `trainable_global_rescale_shift` to `global_rescale_shift_trainble`
- Renamed `PerSpeciesScaleShift_` to `per_species_rescale`
- Change default and allowed values of `metrics_key` from `loss` to `validation_loss`. The old default `loss` will no longer be accepted.
- Renamed `per_species_rescale_trainable` to `per_species_rescale_scales_trainable` and `per_species_rescale_shifts_trainable`

### Fixed
- The first 20 epochs/calls of inference are no longer painfully slow for recompilation
- Set global options like TF32, dtype in `nequip-evaluate`
- Avoid possilbe race condition in caching of processed datasets across multiple training runs

### Removed
- Removed `allowed_species`
- Removed `--update-config`; start a new training and load old state instead
- Removed dependency on `pytorch_geometric`
- `nequip-train` no longer prints the full config, which can be found in the training dir as `config.yaml`.
- `nequip.datasets.AspirinDataset` & `nequip.datasets.WaterDataset`
- Dependency on `pytorch_scatter`

## [0.3.3] - 2021-08-11
### Added
- `to_ase` method in `AtomicData.py` to convert `AtomicData` object to (list of) `ase.Atoms` object(s)
- `SequentialGraphNetwork` now has insertion methods
- `nn.SaveForOutput`
- `nequip-evaluate` command for evaluating (metrics on) trained models
- `AtomicData.from_ase` now catches `energy`/`energies` arrays

### Changed
- Nonlinearities now specified with `e` and `o` instead of `1` and `-1`
- Update interfaces for `torch_geometric` 1.7.1 and `e3nn` 0.3.3
- `nonlinearity_scalars` now also affects the nonlinearity used in the radial net of `InteractionBlock`
- Cleaned up naming of initializers

### Fixed
- Fix specifying nonlinearities when wandb enabled
- `Final` backport for <3.8 compatability
- Fixed `nequip-*` commands when using `pip install`
- Default models rescale per-atom energies, and not just total
- Fixed Python <3.8 backward compatability with `atomic_save`

## [0.3.2] - 2021-06-09
### Added
- Option for which nonlinearities to use
- Option to save models every *n* epochs in training
- Option to specify optimization defaults for `e3nn`

### Fixed
- Using `wandb` no longer breaks the inclusion of special objects like callables in configs

## [0.3.1]
### Fixed
- `iepoch` is no longer off-by-one when restarting a training run that hit `max_epochs`
- Builders, and not just sub-builders, use the class name as a default prefix
### Added
- `early_stopping_xxx` arguments added to enable early stop for platued values or values that out of lower/upper bounds.

## [0.3.0] - 2021-05-07
### Added
- Sub-builders can be skipped in `instantiate` by setting them to `None`
- More flexible model initialization
- Add MD w/ Nequip-ASE-calculator + run-MD script w/ custom Nose-Hoover

### Changed
- PBC must be explicit if a cell is provided
- Training now uses atomic file writes to avoid corruption if interupted
- `feature_embedding` renamed to `chemical_embedding` in default models

### Fixed
- `BesselBasis` now works on GPU when `trainable=False`
- Dataset `extra_fixed_fields` are now added even if `get_data()` returns `AtomicData` objects

## [0.2.1] - 2021-05-03
### Fixed
- `load_deployed_model` now correctly loads all metadata

## [0.2.0] - 2021-04-30
