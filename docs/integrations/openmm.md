# OpenMM

NequIP framework models can be used in [OpenMM](https://openmm.org/) through [OpenMM-ML](https://github.com/openmm/openmm-ml).
See the [OpenMM documentation](https://openmm.org/documentation) and the upstream [OpenMM-ML user guide](https://openmm.github.io/openmm-ml/dev/userguide.html#id4) for installation and usage details.

## Current status in NequIP docs

- The current OpenMM path is based on NequIP [checkpoint files](../guide/getting-started/files.md#checkpoint-files) (`.ckpt`).
- Support for NequIP [package files](../guide/getting-started/files.md#package-files) (`.nequip.zip`) is a work in progress.
- If you need model compilation and/or [GPU kernel modifiers](../guide/accelerations/gpu_kernel_modifiers.md) for this integration path, please open a [GitHub issue](https://github.com/mir-group/nequip/issues).
