# CuEquivariance Acceleration

[CuEquivariance](https://github.com/NVIDIA/cuEquivariance), developed by NVIDIA, provides GPU-accelerated tensor product operations for equivariant neural networks.
This integration accelerates NequIP models during inference, with training support currently a work in progress.

```{warning}
`float64` model compatibility with CuEquivariance remains **untested**. For double precision models (i.e., `model_dtype: float64`), compatibility is not guaranteed.
```

**Requirements:**

- [PyTorch](https://pytorch.org/) >= 2.6
- CUDA-compatible GPU
- [cuequivariance](https://github.com/NVIDIA/cuEquivariance) >= 0.6.0 library installed:

```bash
pip install cuequivariance-torch cuequivariance-ops-torch-cu12
```

## ASE Inference with CuEquivariance

For ASE inference, you can compile your trained model with CuEquivariance acceleration enabled using [`nequip-compile`](../getting-started/workflow.md#compilation):

### ASE AOT Inductor Compilation

```bash
nequip-compile \
    path/to/model.ckpt \
    path/to/compiled_model.nequip.pt2 \
    --device cuda \
    --mode aotinductor \
    --target ase \
    --modifiers enable_CuEquivariance
```

### ASE TorchScript Compilation

```bash
nequip-compile \
    path/to/model.ckpt \
    path/to/compiled_model.nequip.pth \
    --device cuda \
    --mode torchscript \
    --target ase \
    --modifiers enable_CuEquivariance
```

To use the compiled model, you must import `cuequivariance_torch` before loading:

```python
import cuequivariance_torch
from nequip.ase import NequIPCalculator

# Load the compiled model
calc = NequIPCalculator.from_compiled_model(
    "path/to/compiled_model.nequip.pt2/pth",
    device="cuda"
)

# Use with ASE atoms object
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## LAMMPS ML-IAP Integration

CuEquivariance can also be used with LAMMPS through the [ML-IAP interface](../../integrations/lammps/mliap.md).
This provides a stable integration path for production molecular dynamics simulations with CuEquivariance acceleration.

To prepare a model for LAMMPS ML-IAP with CuEquivariance:

```bash
nequip-prepare-lmp-mliap \
  /path/to/model_file \
  /path/to/output.nequip.lmp.pt \
  --modifiers enable_CuEquivariance
```

Where `model_file` can be either a [checkpoint file](../getting-started/files.md#checkpoint-files) (`.ckpt`) or [package file](../getting-started/files.md#package-files).
The resulting `.nequip.lmp.pt` file can be used directly in LAMMPS scripts with the `pair_style mliap` command.
See the [ML-IAP documentation](../../integrations/lammps/mliap.md) for complete usage instructions and examples.