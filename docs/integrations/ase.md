ASE
===

The [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) is a popular Python package providing a framework for working with atomic data, reading and writing common formats, and running various simulations and calculations.

The `nequip` package provides a ASE "calculator" object that allows NequIP and Allegro models to be used through the standardized ASE interface. A calculator can be constructed from a **deployed** model as follows:
```python
from nequip.ase import NequIPCalculator

calculator = NequIPCalculator.from_deployed_model(
    model_path="path/to/deployed_model.pth",
    device="cpu",  # "cuda", etc.
)
```
If you do this, you will see this warning:
```txt
Trying to use chemical symbols as NequIP type names; this may not be correct for your model! To avoid this warning, please provide `species_to_type_name` explicitly
```
ASE models the types of atoms with their atomic numbers, or correspondingly, chemical symbols. `nequip`, on the other hand, can handle an arbitrary number of atom types with arbitrary alphanumeric names (see [Datasets](../guide/dataset.md) for details). By default, `nequip` assumes (as with when the types are specified to `nequip` with the `chemical_symbols` option) that the `nequip` model's types are named after chemical symbols, and maps the atoms from ASE accordingly.  If this is not the case, or if you want to silence this warning, explicitly provide the type mapping from chemical species in ASE to the `nequip` type names:
```python
calculator = NequIPCalculator.from_deployed_model(
    model_path="path/to/deployed_model.pth",
    device="cpu",  # "cuda", etc.
    species_to_type_name={"H": "myHydrogen", "C": "someCarbonType"}
)
```
```{warning}
If you are running MD with custom species, please make sure to set the correct masses in ASE.
```

ASE also enforces a consistent scheme of units (see TODO); `nequip` does not (see [Units](../guide/conventions.md#units)). If your model is not in units of eV and Ångstrom, you must provide conversion factors to bring the model's predictions into ASE's units:
```python
calculator = NequIPCalculator.from_deployed_model(
    model_path="path/to/deployed_model.pth",
    device="cpu",  # "cuda", etc.
    energy_units_to_eV: float = 1.0,
    length_units_to_A: float = 1.0,
)
```

Finally, you may see warnings of the form
```txt
Setting the GLOBAL value for ...
```
`nequip` manages a number of global configuration settings of PyTorch and e3nn and correctly restores those values when a deployed model is loaded. These settings, however, must be set at a global level and thus may affect a host application;  for this reason `nequip` by default will warn whenever overriding global configuration options.  If you need to, these warnings can be silenced with `set_global_options=True`.  (Setting `set_global_options=False` is **strongly** discouraged and might lead to strange issues or incorrect numerical results.)