# `pair_nequip_allegro`

The [`pair_nequip_allegro`](https://github.com/mir-group/pair_nequip_allegro) repository provides pair styles to use NequIP framework interatomic potentials directly in LAMMPS.
This repository contains:

- `pair_nequip` for the NequIP message-passing GNN model
- `pair_allegro` for the strictly local Allegro model


## Model Compilation

[`nequip-compile`](../../guide/getting-started/workflow.md#compilation) is the command used to compile a model (either from a checkpoint file or a package file) for production simulations with the pair styles.
There are two compiler modes: `torchscript` and `aotinductor`, which produce compiled model files with extensions `.nequip.pth` and `.nequip.pt2` respectively.
`nequip-compile` should be performed on the same machine that the LAMMPS simulation will be run on.

To compile a model with TorchScript:
```bash
nequip-compile \
  path/to/ckpt_file/or/package_file \
  path/to/compiled_model.nequip.pth \
  --device [cpu|cuda] \
  --mode torchscript
```

To compile a model with AOTInductor:
```bash
nequip-compile \
  path/to/ckpt_file/or/package_file \
  path/to/compiled_model.nequip.pt2 \
  --device [cpu|cuda] \
  --mode aotinductor \
  --target [pair_nequip|pair_allegro]
```

## Installation and Usage

For detailed installation instructions, compilation of LAMMPS with the pair styles, and usage examples, please refer to the README in the [`pair_nequip_allegro`](https://github.com/mir-group/pair_nequip_allegro) repository.