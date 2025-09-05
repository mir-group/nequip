# ASE

## Introduction
The [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) is a popular Python package providing a framework for working with atomic data, reading and writing common formats, and running various simulations and calculations. 
INequIP provides the {class}`~nequip.ase.NequIPCalculator` for integration with ASE. See the [ASE calculator API](../api/ase.rst) for detailed documentation.

## Creating an ASE Calculator

To use a NequIP framework model with ASE, you need to follow these steps:

1. **Start with a trained model**: You can begin with either a [checkpoint file](../guide/getting-started/files.md#checkpoint-files) (`.ckpt`) from training or a [packaged model file](../guide/getting-started/files.md#package-files) (`.nequip.zip`). See the [file types documentation](../guide/getting-started/files.md) for more details about these formats.

2. **Compile the model for ASE**: Use `nequip-compile` with the `--target ase` flag to create a compiled model suitable for ASE integration:

   ```bash
   nequip-compile \
     path/to/model.ckpt \
     path/to/compiled_model.nequip.pt2 \
     --device cuda \  # or "cpu"
     --mode aotinductor \
     --target ase
   ```

   The device specified during compilation should match the device you'll use with the calculator. For more details about compilation options and requirements, see the [compilation workflow documentation](../guide/getting-started/workflow.md#compilation).

3. **Create the ASE calculator**: Build an ASE {class}`~nequip.ase.NequIPCalculator` from the compiled model file:

```python
from nequip.ase import NequIPCalculator

calculator = NequIPCalculator.from_compiled_model(
    compile_path="path/to/compiled_model.nequip.pt2",
    device="cuda",  # or "cpu"
)
```

### Mapping types from NequIP to ASE
As can be seen in the [ASE calculator API](../api/ase.rst), the  `chemical_symbols` argument is optional. ASE models the types of atoms with their atomic numbers, or correspondingly, chemical symbols. The NequIP framework, on the other hand, can handle an arbitrary number of atom types with arbitrary alphanumeric names. If `chemical_symbols` is not specified, by default, `nequip` assumes that the `nequip` model's types (see [nequip.model](../api/model.rst)) are named after chemical symbols, and maps the atoms from ASE accordingly.

If this is not the case, or if you want to silence the warning from not providing `chemical_symbols`, then explicitly provide `chemical_symbols`, either as list of `nequip` type names or the type mapping from chemical species in ASE to the `nequip` type names:

```python
from nequip.ase import NequIPCalculator

calculator = NequIPCalculator.from_compiled_model(
    compile_path="path/to/compiled_model.nequip.pt2",
    device="cuda",  # or "cpu"
    chemical_symbols={"H": "myHydrogen", "C": "someCarbonType"}
)
```

### Units from NequIP to ASE
The ASE convention uses eV energy units and Å length units while the NequIP framework follows the ([internally consistent](../guide/reference/faq.md#units)) units of the underlying dataset. If it is necessary to account for units conversions, users should specify conversion factors with the arguments `energy_units_to_eV` and `length_units_to_A` (see [ASE calculator API](../api/ase.rst)).

## Example Usage
The NequIP ASE calculator can then be used for standard ASE operations. Below are a few (nonexhaustive) common examples.

### Energy-Volume Curve
Here we use the NequIP model trained in the tutorial to compute energies and forces on various structures to create an energy volume curve.

```python
from ase.build import bulk
import numpy as np
import matplotlib.pyplot as plt
from nequip.ase import NequIPCalculator
import torch

# Initialize the nequip calculator
calculator = NequIPCalculator.from_compiled_model(
    compile_path="path/to/compiled_model.nequip.pt2", 
    chemical_symbols=["Si"], 
    device="cuda" if torch.cuda.is_available() else "cpu",  
)  # use GPUs if available

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

### Structural Relaxations
Here we use a NequIP model to perform structural relaxations, including volume relaxation and tracking the force magnitudes at each ionic step (to catch exploding/hanging relaxations).

```python
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.filters import ExpCellFilter, FrechetCellFilter
import ase.optimize as opt
from pymatgen.core.structure import Structure
from monty.serialization import loadfn
from tqdm import tqdm

from nequip.ase import NequIPCalculator


compile_path = "path/to/compiled_model.nequip.pt2"  # path to compiled model
# choose ASE force optimizer and 'filter' (volume relaxation) methods:
ase_optimizer = "GOQN" # faster than "FIRE" from tests; see SI of https://arxiv.org/abs/2412.19330
# note that GOQN is particularly fast with the optimizer updates here: https://gitlab.com/ase/ase/-/merge_requests/3570
ase_filter: Literal["frechet", "exp"] = "frechet"  # recommended filter
max_steps = 500  # max ionic steps in relaxations
force_max = 0.05  # run until the forces are smaller than this in eV/A

# Initialize Nequip ASE Calculator from checkpoint
calculator = NequIPCalculator.from_compiled_model(
    compile_path=compile_path, 
    device="cuda" if torch.cuda.is_available() else "cpu",
)  # use GPUs if available

# load list of structures to relax:
list_of_structs_to_relax = loadfn("list_of_structs_to_relax.json")  # list of structures previously saved with dumpfn()

relax_results: dict[str, dict[str, Any]] = {}  # dictionary of relaxation results to store

filter_cls: Callable[[Atoms], Atoms] = {  # ASE relaxation filter to use; frechet recommended
    "frechet": FrechetCellFilter,
    "exp": ExpCellFilter,
}[ase_filter]
optimizer_dict = {
    "GPMin": opt.GPMin,
    "GOQN": opt.GoodOldQuasiNewton,
    "BFGSLineSearch": opt.BFGSLineSearch,
    "QuasiNewton":  opt.BFGSLineSearch,
    "SciPyFminBFGS": opt.sciopt.SciPyFminBFGS,
    "BFGS": opt.BFGS,
    "LBFGSLineSearch": opt.LBFGSLineSearch,
    "SciPyFminCG": opt.sciopt.SciPyFminCG,
    "FIRE2": opt.fire2.FIRE2,
    "FIRE": opt.fire.FIRE,
    "LBFGS": opt.LBFGS,
}
optim_cls: Callable[..., opt.optimize.Optimizer] = optimizer_dict[ase_optimizer]  # select ASE optimizer

for struct in tqdm(list_of_structs_to_relax, desc="Relaxing"):
    try:
        atoms = struct.get_ase_atoms()
        atoms.calc = calculator
        if max_steps > 0:
            atoms = filter_cls(atoms)
            with optim_cls(atoms, logfile="/dev/null") as optimizer:
                for _ in optimizer.irun(fmax=force_max, steps=max_steps):
                    forces = atoms.get_forces()
                    if np.max(np.linalg.norm(forces, axis=1)) > 1e6:  # break relaxations which explode, to avoid hanging
                        raise RuntimeError("Forces are exorbitant, exploding relaxation!")

        energy = atoms.get_potential_energy()  # relaxed energy
        # if max_steps > 0, atoms is wrapped by filter_cls, so extract with getattr
        relaxed_struct = Structure.from_ase_atoms(getattr(atoms, "atoms", atoms))
        relax_results[relaxed_struct.formula] = {"structure": relaxed_struct, "energy": energy}
    except Exception as exc:
        print(f"Failed to relax {struct.formula}: {exc!r}")
        continue

relaxed_df = pd.DataFrame(relax_results).T
relaxed_df.to_csv("relaxed_structures.csv")
```