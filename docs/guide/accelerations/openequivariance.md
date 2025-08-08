# OpenEquivariance Acceleration

[OpenEquivariance](https://github.com/PASSIONLab/OpenEquivariance), presented in ["An Efficient Sparse Kernel Generator for O(3)-Equivariant Deep Networks"](https://arxiv.org/abs/2501.13986), is an open-source GPU kernel generator for Clebsch-Gordon tensor products in rotation-equivariant deep neural networks.
It provides up to an order of magnitude acceleration over standard implementations by generating fast GPU kernels for tensor product operations.

```{important}
OpenEquivariance only works with {func}`~nequip.model.NequIPGNNModel`.
```

**Requirements:**

- [PyTorch](https://pytorch.org/) >= 2.4
- CUDA-compatible GPU (NVIDIA or AMD with HIP support)
- [OpenEquivariance](https://github.com/PASSIONLab/OpenEquivariance) library installed, i.e. `pip install openequivariance`
- GCC 9+ for compilation

## Training with OpenEquivariance

To enable OpenEquivariance acceleration during training, use the {func}`~nequip.model.modify` wrapper in your configuration file. This applies the OpenEquivariance modifier to replace standard tensor product operations with optimized GPU kernels:

```yaml
training_module:
  _target_: nequip.train.EMALightningModule
  
  # ... other training module configurations ...
  
  model:
    _target_: nequip.model.modify
    modifiers:
      - modifier: enable_OpenEquivariance
    model:
      _target_: nequip.model.NequIPGNNModel
      seed: 123
      model_dtype: float32
      type_names: [C, H, O, Cu]
      r_max: 5.0
      num_layers: 4
      l_max: 2
      num_features: 32
      # ... your standard model configuration ...
```

The {func}`~nequip.model.modify` function wraps your base model and applies the specified modifiers. The `enable_OpenEquivariance` modifier serves as a drop-in replacement for standard tensor product operations, utilizing JIT kernel generation for optimal performance.

OpenEquivariance composes with [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html), and can be used in conjunction with [train-time compilation](pt2_compilation.md), provided the NequIP version is at least v0.12.0 and OpenEquivariance installation is at least v0.3.0.

## Inference with OpenEquivariance

OpenEquivariance is supported for inference with:
- **[ASE](../../integrations/ase.md)** via TorchScript compilation using [`nequip-compile`](../getting-started/workflow.md#compilation) (AOT Inductor support is work in progress)
- **[LAMMPS](../../integrations/lammps/index.md)** via [ML-IAP integration](../../integrations/lammps/mliap.md)

### ASE-TorchScript Integration

First, compile your model to TorchScript with OpenEquivariance support:

```bash
nequip-compile /path/to/model.ckpt /path/to/compiled_model.nequip.pth \
  --mode torchscript \
  --device cuda \
  --target ase \
  --modifiers enable_OpenEquivariance
```

To use the compiled model with ASE, you must explicitly import OpenEquivariance in your Python script before loading the model:

```python
import openequivariance  # Required: must import before loading compiled model
from nequip.ase import NequIPCalculator

# Load the compiled model
calc = NequIPCalculator.from_compiled_model(
    "/path/to/compiled_model.nequip.pth",
    device="cuda"
)

# Use with ASE atoms object
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

If `openequivariance` is not imported before model loading, you will encounter this error:
```
RuntimeError: Couldn't resolve type '{}', did you forget to add its build dependency?__torch__.torch.classes.libtorch_tp_jit.TorchJITConv
```

### LAMMPS ML-IAP Integration

OpenEquivariance can also be used with LAMMPS through the [ML-IAP interface](../../integrations/lammps/mliap.md).
This provides a stable integration path for production molecular dynamics simulations with OpenEquivariance acceleration.

To prepare a model for LAMMPS ML-IAP with OpenEquivariance:

```bash
nequip-prepare-lmp-mliap \
  /path/to/model_file \
  /path/to/output.nequip.lmp.pt \
  --modifiers enable_OpenEquivariance
```

Where `model_file` can be either a [checkpoint file](../getting-started/files.md#checkpoint-files) (`.ckpt`) or [package file](../getting-started/files.md#package-files).
The resulting `.nequip.lmp.pt` file can be used directly in LAMMPS scripts with the `pair_style mliap` command.
See the [ML-IAP documentation](../../integrations/lammps/mliap.md) for complete usage instructions and examples.
