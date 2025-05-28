# Data Statistics, Loss, and Metrics

Data statistics, loss functions, and metrics are all configured by specifying a field (e.g. `total_energy`, `forces`, [etc.](../api/data_fields.rst)) and a quantity to calculate for it. Data statistics are bare quantities (e.g. `Mean`, `RootMeanSquare`, `Max`, etc), while loss and metrics use error quantities (e.g. `MeanSquaredError`, `MeanAbsoluteError`, etc).

## Data Statistics

There are two key reasons why one would use the automatic data statistics utilities during `nequip-train`: for inspection to get a feel for the data distribution, and to use specific data statistics as [model parameters](model.md/#training-data-statistics-as-hyperparameters) during model initialization.

Data statistics is configured by setting the `stats_manager` argument of the {class}`~nequip.data.datamodule.NequIPDataModule` used (under the `data` [block of the config](config.md/#data)). 
The `stats_manager` refers to a {class}`~nequip.data.DataStatisticsManager` object (see docs [here](../api/data_stats.rst)).
Follow the link to the {class}`~nequip.data.DataStatisticsManager` API to learn more about how to use it.
There is an example of the full config block for a base {class}`~nequip.data.DataStatisticsManager` object.
A simplified and less verbose wrapper is provided through the {class}`~nequip.data.CommonDataStatisticsManager` for typical use cases where one's data contains `total_energy` and `forces` (and `edge_index`, i.e. the neighborlist that will be computed on-the-fly).

## Loss and Metrics

All loss components and metrics are in the physical units associated with the dataset.  For example, if the dataset uses force units of eV/Å, a force mean-squared error (MSE) would have units of (eV/Å)².
  
Both the loss function and metrics used for validation and testing are configured through {class}`~nequip.train.MetricsManager` objects, as an argument to the `training_module` [block in the config](config.md/#training_module).
Details are provided in the {class}`~nequip.train.MetricsManager` [API page](../api/metrics.rst).
Simplified wrappers are provided for typical use cases, i.e. the {class}`~nequip.train.EnergyForceLoss` and {class}`~nequip.train.EnergyForceStressLoss` loss function wrappers, and the {class}`~nequip.train.EnergyForceMetrics` and {class}`~nequip.train.EnergyForceStressMetrics` metrics wrappers.

In addition to the individual metrics specified, the {class}`~nequip.train.MetricsManager` also computes a `weighted_sum` based on the `coeff` (coefficient) for each metric.
For loss functions, this quantity is used as the loss function as a weighted sum of specified loss components.
For metrics, the weighted sum could be useful for accounting for energy-force(-stress) balancing for monitoring.
For example, `val0_epoch/weighted_sum` can be monitored and used to condition the behavior of learning rate scheduling or early stopping.
