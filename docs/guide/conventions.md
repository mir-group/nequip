# Conventions

## Data Format Conventions
 - Cell vectors are given in ASE style as the **rows** of the cell matrix

## Pressure / stress / virials

`nequip` always expresses stress in the "consistent" units of `energy / length³`, which are **not** the typical physical units used by many codes for stress. For example, consider data where the cell and atomic positions are provided in Å and the energy labels are in eV but the stress is given in GPa or kbar. The user would then need to convert the stress to eV/Å³ before providing the data to `nequip`.

```{warning}
Training labels for stress in the original dataset must be pre-processed by the user to be in consistent units.
```

Stress also includes an arbitrary sign convention, for which we adopt the choice that `virial = -stress x volume  <=>  stress = (-1/volume) * virial`.
This is the most common sign convention in the literature (e.g. adopted by `ASE`), but notably differs from that used by VASP (see [here](https://www.vasp.at/wiki/index.php/ISIF)).
In the NequIP convention, stress is defined as the derivative of the energy $E$ with respect to the strain tensor $\eta_{ji}$, divided by the volume $\Omega$:

$$\sigma_{ij} = \frac{1}{\Omega} \frac{\delta E} {\delta \eta_{ji}}$$

Positive diagonal entries of the stress tensor imply that the system is _under tensile strain_ and wants to compress, while negative values imply that the system is _under compressive strain_ and wants to expand. When VASP results are parsed by `ASE`, the sign is flipped to match the `nequip` convention.

If your dataset uses the opposite sign convention, either preprocess the dataset to be in the NequIP stress sign convention, or use the {class}`~nequip.data.transforms.StressSignFlipTransform` data transform.

Users that have data with virials but seek to train on stress can use the data transform {class}`~nequip.data.transforms.VirialToStressTransform`.
