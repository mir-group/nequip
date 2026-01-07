# torch-sim

## Introduction
[torch-sim](https://github.com/TorchSim/torch-sim) is an atomistic simulation engine built in PyTorch that supports automatic batching and GPU memory management for machine learning potentials. NequIP provides the {class}`~nequip.integrations.torchsim.NequIPTorchSimCalc` for integration with torch-sim.

## Installation

**Note:** torch-sim is not included with NequIP and must be installed separately. It is recommended to install torch-sim from source:

```bash
git clone https://github.com/TorchSim/torch-sim.git
cd torch-sim
pip install -e .
```

## Creating a torch-sim Calculator

To use a NequIP framework model with torch-sim:

1. **Start with a trained model**: Begin with either a [checkpoint file](../guide/getting-started/files.md#checkpoint-files) (`.ckpt`) from training or a [packaged model file](../guide/getting-started/files.md#package-files) (`.nequip.zip`).

2. **Compile the model for torch-sim**: Use `nequip-compile` with the `--target batch` flag to create a compiled model suitable for batched evaluation:

   ```bash
   nequip-compile \
     path/to/model.ckpt \
     path/to/compiled_model.nequip.pt2 \
     --device cuda \  # or "cpu"
     --mode aotinductor \
     --target batch
   ```

   The `aotinductor` mode is recommended for better performance but requires PyTorch 2.6+. Alternatively, `torchscript` compilation (producing `.nequip.pth` files) also works if `aotinductor` is unavailable. The device specified during compilation should match the device you'll use with the calculator. For more details about compilation options, see the [compilation workflow documentation](../guide/getting-started/workflow.md#compilation).

3. **Create the torch-sim calculator**: Build a {class}`~nequip.integrations.torchsim.NequIPTorchSimCalc` from the compiled model file:

```python
from nequip.integrations.torchsim import NequIPTorchSimCalc

calculator = NequIPTorchSimCalc.from_compiled_model(
    compile_path="path/to/compiled_model.nequip.pt2",
    device="cuda",  # or "cpu"
)
```

### Mapping types from NequIP to torch-sim
The NequIP framework can handle arbitrary alphanumeric atom type names.
The `chemical_species_to_atom_type_map` argument controls how atomic numbers from torch-sim structures map to the model's atom types.

**Default behavior (with warning):** If not specified, the calculator assumes model type names are chemical symbols and uses an identity mapping. A warning is issued to alert you of this assumption:

```python
calculator = NequIPTorchSimCalc.from_compiled_model(
    compile_path="path/to/compiled_model.nequip.pt2",
    device="cuda",
    # Omitting chemical_species_to_atom_type_map triggers a warning
)
```

**Explicit identity mapping (no warning):** When you know the model type names correspond exactly to chemical species, set `chemical_species_to_atom_type_map=True` to silence the warning:

```python
calculator = NequIPTorchSimCalc.from_compiled_model(
    compile_path="path/to/compiled_model.nequip.pt2",
    device="cuda",
    chemical_species_to_atom_type_map=True  # identity mapping, no warning
)
```

**Custom mapping:** If the model uses non-standard type names (e.g., charge states, coarse-grained types), provide an explicit mapping dict:

```python
calculator = NequIPTorchSimCalc.from_compiled_model(
    compile_path="path/to/compiled_model.nequip.pt2",
    device="cuda",
    chemical_species_to_atom_type_map={"H": "H+", "C": "C_sp3", "O": "O-"}
)
```

## Example Usage

### Batched Evaluation
The primary use case for torch-sim integration is efficient batched evaluation of multiple systems. Here's a minimal example:

```python
import torch
import torch_sim as ts
from ase.build import bulk
from nequip.integrations.torchsim import NequIPTorchSimCalc

# Initialize the calculator
calculator = NequIPTorchSimCalc.from_compiled_model(
    compile_path="path/to/compiled_model.nequip.pt2",
    device="cuda" if torch.cuda.is_available() else "cpu",
    chemical_species_to_atom_type_map=True,  # identity mapping
)

# Create multiple structures
structures = [
    bulk("Si", crystalstructure="diamond", a=5.43 * scale, cubic=True)
    for scale in [0.98, 1.00, 1.02]
]

# Convert to torch-sim state and evaluate in a single batched call
sim_state = ts.io.atoms_to_state(structures, device="cuda", dtype=torch.float64)
results = calculator(sim_state)

# Access results
energies = results["energy"]  # shape: (n_systems,)
forces = results["forces"]    # shape: (n_atoms_total, 3)
stress = results["stress"]    # shape: (n_systems, 3, 3)
```
