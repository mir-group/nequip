# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Most recent change on the bottom.

## [Unreleased]

## [0.3.0] - 2021-05-07
### Added
- More flexible model initialization
- Add MD w/ Nequip-ASE-calculator + run-MD script w/ custom Nose-Hoover

### Changed
- Training now uses atomic file writes to avoid corruption if interupted
- `feature_embedding` renamed to `chemical_embedding` in default models

### Fixed
- Dataset `extra_fixed_fields` are now added even if `get_data()` returns `AtomicData` objects

## [0.2.1] - 2021-05-03
### Fixed
- `load_deployed_model` now correctly loads all metadata

## [0.2.0] - 2021-04-30
