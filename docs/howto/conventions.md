# Conventions and units

## Conventions
 - Cells vectors are given in ASE style as the **rows** of the cell matrix
 - The first index in an edge tuple (``edge_index[0]``) is the center atom, and the second (``edge_index[1]``) is the neighbor

## Units

`nequip` has no prefered system of units; models, errors, predictions, etc. will always be in the units of the original dataset used.

```{warning}
`nequip` cannot and does not check the consistency of units in inputs you provide, and it is your responsibility to ensure consistent treatment of input and output units
```

Losses (`training_loss_f`, `validation_loss_e`, etc.) do **not** have physical units. Errors (`training_f_rmse`, `validation_f_rmse`) are always reported in physical units.

## Pressure / stress / virials

`nequip` always expresses stress in the "consistent" units of `energy / length^3`, which are **not** the typical physical units used by many codes for stress.

```{warning}
Training labels for stress in the original dataset must be pre-processed by the user to be in consistent units.
```

Stress also includes an arbitrary sign convention, for which we adopt the choice that `virial = -stress x volume  <=>  stress = (-1/volume) * virial`.

```{warning}
Training labels for stress in the original dataset must be pre-processed by the user to be in **this sign convention**, which they may or may not already be depending on their origin.
```