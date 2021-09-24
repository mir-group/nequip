# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Most recent change on the bottom.

## [Unreleased]
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

### Changed
- Name processed datasets based on a hash of their parameters to ensure only valid cached data is used
- Do not use TensorFloat32 by default on Ampere GPUs until we understand it better
- No atomic numbers in networks
- `dataset_energy_std`/`dataset_energy_mean` to `dataset_total_energy_*`

### Fixed
- The first 20 epochs/calls of inference are no longer painfully slow for recompilation

### Removed
- Removed `allowed_species`
- Removed `--update-config`; start a new training and load old state instead

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
