# Loss functions and metrics

## Loss functions
The loss function is defined by the `loss_coeffs` option, which describes a weighted sum of loss terms. At simplest, it specifies a single field on which to apply a mean-squared error loss:
```yaml
loss_coeffs: forces
```
It can also combine loss functions on multiple fields:
```yaml
loss_coeffs:
  forces: 1.0
  total_energy: 10.0
```
(The numbers are the relative weights.)

The loss term's functional form can also be specified (by default `MSELoss`):
```yaml
loss_coeffs:
  total_energy:
    - 1.0.0
    - MSELoss
```
Other options include `L1Loss`, and the `PerAtom` and `PerSpeices` prefixes.

The `PerAtom` prefix normalizes the error for each per-frame label by the number of atoms in that frame, and thus helps to normalize the scale of errors on size-extensive quantities such as the total energy (see section TODO of TODO):
```yaml
loss_coeffs:
  total_energy:
    - 1.0
    - PerAtomMSELoss
```

The `PerSpecies` prefix averages the error on a per-atom quantity within each species of atom, and then takes an average over all species averages. This can help re-balance systems with extreme imbalance between the number of atoms of different species, such as in the catalytic system (formate dehydrogenation on a copper surface) described in [1]:
```yaml
loss_coeffs:
  forces:
    - 1.0
    - PerSpeciesL1Loss
```

For typical potential fitting applications, we recommend starting with:
```yaml
loss_coeffs:
  forces: 1.0
  total_energy:
    - 1.0
    - PerAtomMSELoss
```
if forces and energies are available,
```yaml
loss_coeffs:
  forces: 1.0
  total_energy:
    - 1.0
    - PerAtomMSELoss
  virial:
    - 1.0
    - PerAtomMSELoss
```
if forces, energies, and virials are available, and
```yaml
loss_coeffs:
  forces: 1.0
  total_energy:
    - 1.0
    - PerAtomMSELoss
  stress: 1.0
```
if forces, energies, and stress tensors are available.
Virial and stress tensors contain equivalent information and should **not** both be used in the same loss function; because the virial is size extensive but the stress is not, we use a `PerAtom` loss only with the virial.

## Metrics

**References:**
 1. todo Nequip
 2. todo Allergo