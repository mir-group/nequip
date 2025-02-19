# Conventions

## Data Format Conventions
 - Cells vectors are given in ASE style as the **rows** of the cell matrix
 - The first index in an edge tuple (``edge_index[0]``) is the center atom, and the second (``edge_index[1]``) is the neighbor


## Pressure / stress / virials

`nequip` always expresses stress in the "consistent" units of `energy / length³`, which are **not** the typical physical units used by many codes for stress. For example, consider data where the cell and atomic positions are provided in Å and the energy labels are in eV but the stress is given in GPa or kbar. The user would then need to convert the stress to eV/Å³ before providing the data to `nequip`.

```{warning}
Training labels for stress in the original dataset must be pre-processed by the user to be in consistent units.
```

Stress also includes an arbitrary sign convention, for which we adopt the choice that `virial = -stress x volume  <=>  stress = (-1/volume) * virial`. This is the most common sign convention in the literature (e.g. adopted by `ASE`), but notably differs from that used by VASP (see [here](https://www.vasp.at/wiki/index.php/ISIF)). In the sign convention used by `nequip`, stress is defined as the derivative of the energy $E$ with respect to the strain tensor $\eta_{ji}$:

$$\sigma_{ij} = \frac{\delta E} {\delta \eta_{ji}}$$

such that a positive in the diagonals implies the system is _under tensile strain_ and wants to compress, while a negative value implies the system is _under compressive strain_ and wants to expand. When VASP results are parsed by `ASE`, the sign is flipped to match the `nequip` convention.

```{warning}
Training labels for stress in the original dataset must be pre-processed by the user to be in **this sign convention**, which they may or may not already be depending on their origin.
```

Users that have data with virials but seek to train on stress can use the data transform `nequip.data.transforms.VirialToStressTransform`.
