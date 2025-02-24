# ASE

## Introduction
The [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) is a popular Python package providing a framework for working with atomic data, reading and writing common formats, and running various simulations and calculations.

The `nequip` package provides seamless integration of NequIP models with the standard ASE interface through an [ASE Calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html). The `NequIPCalculator` can be constructed from a model [compiled](../guide/workflow.md#compilation) with `nequip-compile` as detailed in the [ASE calculator API](../api/ase.rst). Other options include using a model from a checkpoint file or one that has been [packaged](../guide/workflow.md#packaging), but it is strongly recommended to use compiled models as they are optimized for inference.

## Creating an ASE Calculator

The following code block shows how one can built an ASE `NequIPCalculator` from a compiled model file or checkpoint file.

```python
from nequip.ase import NequIPCalculator

# from compiled model (optimized for inference)
calculator = NequIPCalculator.from_compiled_model(
    compile_path="path/to/compiled_model.nequip.pt2",
    device="cpu",  # "cuda", etc
)

# from checkpoint (not optimized for inference)
calculator = NequIPCalculator.from_checkpoint_model(
    ckpt_path="path/to/checkpoint_model.ckpt",
    device="cpu",  # "cuda", etc.
)
```

### Mapping types from NequIP to ASE
As can be seen in the [ASE calculator API](../api/ase.rst), the  `chemical_symbols` argument is optional. ASE models the types of atoms with their atomic numbers, or correspondingly, chemical symbols. The NequIP framework, on the other hand, can handle an arbitrary number of atom types with arbitrary alphanumeric names. If `chemical_symbols` is not specified, by default, `nequip` assumes that the `nequip` model's types (see [nequip.model](../api/model.rst)) are named after chemical symbols, and maps the atoms from ASE accordingly.

If this is not the case, or if you want to silence the warning from not providing `chemical_symbols`, then explicitly provide `chemical_symbols`, either as list of `nequip` type names or the type mapping from chemical species in ASE to the `nequip` type names:

```python
from nequip.ase import NequIPCalculator

calculator = NequIPCalculator.from_checkpoint_model(
    ckpt_path="path/to/checkpoint_model.ckpt",
    device="cpu",  # "cuda", etc.
    chemical_symbols={"H": "myHydrogen", "C": "someCarbonType"}
)
```

### Units from NequIP to ASE
The ASE convention uses eV energy units and Å length units while the NequIP framework follows the ([internally consistent](../guide/faq_errors.md#units)) units of the underlying dataset. If it is necessary to account for units conversions, users should specify conversion factors with the arguments `energy_units_to_eV` and `length_units_to_A` (see [ASE calculator API](../api/ase.rst)).

## Example Usage
The NequIP ASE calculator can then be used for standard ASE operations. Below are a few (nonexhaustive) common examples.

### Energy-Volume Curve
Here we use the NequIP model trained in the tutorial to compute energies and forces on various structures to create an energy volume curve.

```python
from ase.build import bulk
import numpy as np
import matplotlib.pyplot as plt
from nequip.ase import NequIPCalculator

# Initialize the nequip calculator
calculator = NequIPCalculator.from_checkpoint_model(
    ckpt_path="/content/results/best.ckpt", chemical_symbols=["Si"], device="cpu"
)

# Range of scaling factors for lattice constant
scaling_factors = np.linspace(0.95, 1.05, 10)
volumes = []
energies = []

# Loop through scaling factors, calculate energy, and collect volumes and energies
for scale in scaling_factors:

    # Generate the cubic silicon structure with 216 atoms
    scaled_si = bulk("Si", crystalstructure="diamond", a=5.43 * scale, cubic=True)
    scaled_si *= (3, 3, 3)  # Make a supercell (3x3x3) to get 216 atoms
    scaled_si.calc = calculator

    volume = scaled_si.get_volume()
    energy = scaled_si.get_potential_energy()
    volumes.append(volume)
    energies.append(energy)

# Plot the energy-volume curve
plt.figure(figsize=(8, 6))
plt.plot(volumes, energies, marker="o", label="E-V Curve")
plt.xlabel("Volume (Å³)", fontsize=14)
plt.ylabel("Energy (eV)", fontsize=14)
plt.title("Energy-Volume Curve for Cubic Silicon", fontsize=16)
plt.legend(fontsize=12)
plt.grid()
plt.show()

```