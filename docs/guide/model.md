# Model Hyperparameters

Models are configured by specifying a function, which we usually call the "model builder," and its arguments, which are the hyperparameters of the model. For example:
```yaml
model:
    _target_: nequip.model.NequIPGNNModel  # the model builder
    hyperparam1: 7.7
    hyperparam2: 123
    # ...
```
To see the documentation for individual hyperparameters, look at the Python API documentation for the model builder function ({func}`~nequip.model.NequIPGNNModel` in this example).  Model builders are usually in the `.model` subpackage of `nequip` or a `nequip` extension package.

## Training data statistics as hyperparameters

Sometimes we may want to derive certain hyperparameters from the statistics of the data used for training. For example, it is reasonable to use the average number of neighbors around a central atom to normalize a sum over edge features. Users are able to configure specific types of dataset statistics to compute with the `stats_manager` argument of the {class}`~nequip.data.datamodule.NequIPDataModule` used under the `data` section of the config file. Consider the following example:

```yaml
data:
  _target_:  [the datamodule class]
  
  # datamodule arguments ...

  stats_manager:
    _target_: nequip.data.CommonDataStatisticsManager
    type_names: ${model_type_names}
    dataloader_kwargs:
      batch_size: 10
```
In the above example, we call {class}`~nequip.data.CommonDataStatisticsManager`, which will automatically compute the following dataset statistics: `num_neighbors_mean`, `per_atom_energy_mean`, `forces_rms`, and `per_type_forces_rms`. (One could also configure custom dataset statistics with the [{class}`~nequip.data.DataStatisticsManager`](../api/data_stats).) We are then able to refer to these dataset statistics variables when configuring the model hyperparameters in the `model` section of the config file:
```yaml
training_module:
  _target_: nequip.train.EMALightningModule
  
  # other `EMALightningModule` arguments...

  model:
    _target_: nequip.model.NequIPGNNModel

    # other model hyperparameters...

    avg_num_neighbors: ${training_data_stats:num_neighbors_mean}
    per_type_energy_shifts: ${training_data_stats:per_atom_energy_mean}
    per_type_energy_scales: ${training_data_stats:per_type_forces_rms}
```
Note the special syntax `${training_data_stats:name_of_dataset_statistic}` that resembles variable interpolation (it's actually an {mod}`omegaconf` [resolver](https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html)). Under the hood, it tells the NequIP infrastructure to look for the dataset statistics computed by the `stats_manager` and copy their values into the `model` configuration dictionary.

```{tip}
The `name` of the dataset statistic is customizable, but the same `name` must be used when referring to it from the `model` section of the config file. For example, if you call the dataset statistic `name: my_custom_stat`, the model part of the config should use `${training_data_stats:my_custom_stat}`.
```

The three dataset statistics `num_neighbors_mean`, `per_atom_energy_mean`, and `forces_rms` are sufficient for most cases. (Note that it is often advisable to use isolated atom energies as `per_type_energy_shifts` instead of `per_atom_energy_mean`.) The NequIP infrastructure supports other dataset statistics: see [Dataset Statistics](../api/data_stats).

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

For the scales, it is common to use the root mean square forces (which could be calculated per-atom-type) for `per_type_energy_scales`. For energy-only datasets, one could use the total energy's standard deviation.

The section on [training data statistics as hyperparameters](#training-data-statistics-as-hyperparameters) explains how one can compute dataset statistics such as the mean per-atom energy and root mean square forces to be used as model parameters.

```{tip}
`per_type_energy_shifts` and `per_type_energy_scales` can take either a single value or as many values as the number of atom types (given as a dictionary mapping type names to values). When a single value is used, a slightly more optimized code path is taken.
```

Users may also consider toggling the `per_type_energy_shifts_trainable` and `per_type_energy_scales_trainable` hyperparameters (though one may not want to set `per_type_energy_shifts_trainable: true` if one seeks to preserve the expected dissociation behavior to the isolated atom regime based on isolated atom energies one may want to impose).

## Ziegler-Biersack-Littmark (ZBL) Potential

For practical molecular dynamics simulations, it may be favorable to train models with a strong prior for repulsion at close atomic distances. One can add the Ziegler-Biersack-Littmark (ZBL) screened nuclear repulsion term as a `pair_potential` in NequIP and {mod}`allegro` models. This section of the config file could look something like

```yaml
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
`units` refer to LAMMPS unit system names, and can be `metal` (eV and Angstroms) or `real` (kcal/mol and Angstroms). Note that one must also specify `chemical_species`, which are a list of elements in the same order as the `type_names`. Here, the config snippet assumes that the list is defined elsewhere as `chemical_symbols` and is interpolating that variable (since the variable is also needed in configuring the `data` part of the config file).

```{warning}
If you use ZBL, your data must agree with the system of units specified in the `units` parameter.
```