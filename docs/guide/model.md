# Model Hyperparameters


## Dataset Statistics as Parameters

TODO: show data stats manager and explain how the resolver syntax works. Cover`avg_num_neighbors`, `forces_rms`, `per_type_forces_rms`, `per_atom_energy_mean`.

## Energy Shifts & Scales

Both NequIP and Allegro models can be thought of as predicting an **unscaled** per-atom energy contribution $\tilde{E}_i$ where $i$ is an atom index. The per-atom energies are then computed as 

$E_i = \alpha^{t_i} \tilde{E}_i + E_{0}^{t_i}$


where $t_i$ refers to the type or species of atom $i$, $E_{0}^{t_i}$ represents the `per_type_energy_shifts` argument, and $\alpha^{t_i}$ represents the `per_type_energy_scales` argument.

NequIP and Allegro models are physically constrained to ensure that $E_i = E_0^{t_i}$ in the absence of neighboring atoms. Often, it is reasonable to use isolated atom energies (computed by the same reference method as the training data) as the `per_type_energy_shifts`, especially if one requires the model to perform well in the dissociation limit. Other common options is to use the mean per-atom energy, and to perform least squares regression to fit a dataset for per-type energies.

As for scales, it is common to use the root mean square forces (which could be on a per-type basis) for `per_type_energy_scales`. For energy-only datasets, one could use total energy standard deviations.

The section on [Dataset Statistics as Parameters](#dataset-statistics-as-parameters) explains how one can compute dataset statistics such as the mean per-atom energy and root mean square forces to be used as model parameters.

```{tip}
`per_type_energy_shifts` and `per_type_energy_scales` can take either a single value or as many values as the number of atom types (given in the same order as `type_names`). When a single value is used, a slightly optimized code path is taken.
```

Users may also consider toggling `per_type_energy_shifts_trainable` and `per_type_energy_scales_trainable` as tunable hyperparameters (though one may not want to set `per_type_energy_shifts_trainable: true` if one seeks to preserve the expected dissociation behavior to the isolated atom regime based on isolated atom energies one may want to impose).

```{tip}
Regardless of whether `per_type_energy_shifts` and/or `per_type_energy_scales` are specified as a single value or on a per-type basis, making the shifts/scales trainable will always lead to a per-type treatment internally (which may lead to a small additional expense).
```

## Ziegler-Biersack-Littmark (ZBL)

For practical molecular dynamics simulations, it may be favorable to train models with a strong prior for repulsion at close atomic distances. One can add the Ziegler-Biersack-Littmark (ZBL) screened nuclear repulsion term as a `pair_potential` in NequIP and Allegro models. This section of the config file could look something like

```
training_module:
  _target_: nequip.train.EMALightningModule
  
  # other `EMALightningModule` arguments

  model:
    _target_: nequip.model.NequIPGNNModel

    # other model hyperparameters

    pair_potential:
      _target_: nequip.nn.pair_potential.ZBL
      units: metal     
      chemical_species: ${chemical_symbols}   

```
`units` refer to LAMMPS unit names, and can be `metal` (eV and Angstroms) or `real` (kcal/mol and Angstroms). Note that one must also specify `chemical_species`, which are a list of elements in the same order as the `type_names`. Here, the config snippet assumes that the list is defined elsewhere as `chemical_symbols` and is interpolating that variable (since the variable is also needed in configuring the `data` part of the config file).