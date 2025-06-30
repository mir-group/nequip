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

For inference, you need to [`nequip-compile`](../getting-started/workflow.md#compilation) your trained model with OpenEquivariance enabled.
Currently, OpenEquivariance only works with TorchScript compilation for use in [ASE](../../integrations/ase.md) (Atomic Simulation Environment).

### Supported Integrations

| [Compilation Mode](../getting-started/workflow.md#compilation) | [ASE](../../integrations/ase.md) | [LAMMPS](../../integrations/lammps.md) |
|:-------------------------------------------:|:------------------------------:|:-----------------------------------:|
| TorchScript (`.nequip.pth`) | ✅ Stable | 🔨 Work in Progress |
| AOT Inductor (`.nequip.pt2`) | 🔨 Work in Progress | 🔨 Work in Progress |

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
