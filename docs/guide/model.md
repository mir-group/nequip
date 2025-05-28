# Model Hyperparameters


## Dataset Statistics as Parameters

Certain choices of model hyperparameters can be derived from the statistics of the dataset used for training. For example, it is reasonable to use the average number of neighbors around a central atom to normalize the sum over edge features. Users are able to configure specific types of dataset statistics to compute with the `stats_manager` argument of the `DataModule` used under the `data` section of the config file. Let us consider the following example.

```
data:
  _target_: # the datamodule
  
  # datamodule arguments

  stats_manager:
    _target_: _target_: nequip.data.CommonDataStatisticsManager
    type_names: ${model_type_names}
    dataloader_kwargs:
      batch_size: 10
```
In the above example, we call `nequip.data.CommonDataStatisticsManager`, which will automatically compute the following dataset statistics: `num_neighbors_mean`, `per_atom_energy_mean`, `forces_rms`, and `per_type_forces_rms`. One could also configure custom dataset statistics with `nequip.data.DataStatisticsManager`. See [API docs](../../api/data_stats) for the dataset statistics managers for more details. We are then able to refer to these dataset statistics variables when configuring the model hyperparameters in the `model` section of the config file, as follows.
```
training_module:
  _target_: nequip.train.EMALightningModule
  
  # other `EMALightningModule` arguments

  model:
    _target_: nequip.model.NequIPGNNModel

    # other model hyperparameters

    avg_num_neighbors: ${training_data_stats:num_neighbors_mean}
    per_type_energy_shifts: ${training_data_stats:per_atom_energy_mean}
    per_type_energy_scales: ${training_data_stats:per_type_forces_rms}
```
Note the special syntax `${training_data_stats:name_of_dataset_statistic}` that resembles variable interpolation (it's actually an `omegaconf` [resolver](https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html)). Under the hood, it tells the NequIP infrastructure to look for the dataset statistics computed by the `stats_manager` and copy their values into the `model` configuration dictionary.

```{tip}
The `name` of the dataset statistic is customizable, but the same `name` must be used when referring to it from the `model` section of the config file. For example, if you call the dataset statistic `name: my_custom_stat`, the model part of the config should use `${training_data_stats:my_custom_stat}`.
```

The three dataset statistics `num_neighbors_mean`, `per_atom_energy_mean`, and `forces_rms` are often sufficient for most cases (though note that it might be favorable to use isolated atom energies as `per_type_energy_shifts` instead of `per_atom_energy_mean`). The NequIP infrastructure supports other possibilities, and interested users are directed to the [Dataset Statistics page](../api/data_stats) to learn more.

## Energy Shifts & Scales

Both NequIP and Allegro models can be thought of as predicting an **unscaled** per-atom energy contribution $\tilde{E}_i$ where $i$ is an atom index. The per-atom energies are then computed as 

$E_i = \alpha^{t_i} \tilde{E}_i + E_{0}^{t_i}$


where $t_i$ refers to the type or species of atom $i$, $E_{0}^{t_i}$ represents the `per_type_energy_shifts` argument, and $\alpha^{t_i}$ represents the `per_type_energy_scales` argument.

NequIP and Allegro models are physically constrained to ensure that $E_i = E_0^{t_i}$ in the absence of neighboring atoms, that is, the per-atom energies approach the provided (and potentially trained) per-atom shifts in the dissociation limit.
It is recommended to use isolated atom energies (computed by the same reference method as the training data) as the `per_type_energy_shifts`, especially if one requires the model to perform well in the dissociation limit, e.g.
```
  model:
    _target_: nequip.model.NequIPGNNModel
    # other args
    per_type_energy_shifts: 
      C: 1.234
      H: 2.345
      O: 3.456
```

In the absence of isolated atom energies, a common fallback is to use the mean per-atom energy, provided automatically by the dataset statistics utilities, i.e.
```
  per_type_energy_shifts: ${training_data_stats:per_atom_energy_mean}
```

```{warning}
Using the automatically computed mean per-atom energy as the `per_type_energy_shifts` is often a cause of poor training, especially when the isolated atom energies (computed from the method used to construct the training data) of the different atom types vary in scale while the per-atom energy mean is a single value.
Even if the shifts are trainable, it is unlikely for the shifts to change rapidly enough over the course of training to approach the expected isolated atom energies.
```

As for scales, it is common to use the root mean square forces (which could be on a per-type basis) for `per_type_energy_scales`. For energy-only datasets, one could use total energy standard deviations.

The section on [Dataset Statistics as Parameters](#dataset-statistics-as-parameters) explains how one can compute dataset statistics such as the mean per-atom energy and root mean square forces to be used as model parameters.

```{tip}
`per_type_energy_shifts` and `per_type_energy_scales` can take either a single value or as many values as the number of atom types (given in the same order as `type_names`). When a single value is used, a slightly optimized code path is taken.
```

Users may also consider toggling `per_type_energy_shifts_trainable` and `per_type_energy_scales_trainable` as tunable hyperparameters (though one may not want to set `per_type_energy_shifts_trainable: true` if one seeks to preserve the expected dissociation behavior to the isolated atom regime based on isolated atom energies one may want to impose).

```{tip}
Regardless of whether `per_type_energy_shifts` and/or `per_type_energy_scales` are specified as a single value or on a per-type basis, making the shifts/scales trainable will always lead to a per-type treatment internally (which may lead to a small additional expense).
```

## Ziegler-Biersack-Littmark (ZBL) Potential

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