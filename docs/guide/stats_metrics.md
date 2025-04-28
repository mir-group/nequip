# Data Statistics, Loss, and Metrics

Broadly, data statistics, loss functions, and metrics can be configured in similar (but not the same) ways, where one can specify specific fields (e.g. `total_energy`, `forces`, etc. See [here](../api/data_fields.rst) for a complete list.), and corresponding quantities to calculate for them. Data statistics are bare quantities (e.g. `Mean`, `RootMeanSquare`, `Max`, etc), while loss and metrics use error quantities (e.g. `MeanSquaredError`, `MeanAbsoluteError`, etc).

## Data Statistics

There are two key reasons why one would use the automatic data statistics utlities during `nequip-train`: for inspection to get a feel for the data distribution, and to use specific data statistics as[model parameters](model.md/#dataset-statistics-as-parameters) during model initialization.

Data statistics is configured by setting the `stats_manager` argument of the [DataModules](../api/datamodule.rst) used (under the `data` [block of the config](config.md/#data)). 
The `stats_manager` refers to a `DataStatisticsManager` object (see docs [here](../api/data_stats.rst)).
Follow the link to the `DataStatisticsManager` API to learn more about how to use it.
There is an example of the full config block for a base `DataStatisticsManager` object.
A simplified and less verbose wrapper is provided through the `CommonDataStatisticsManager` for typical use cases where one's data contains `total_energy` and `forces` (and `edge_index`, i.e. the neighborlist that will be computed on-the-fly).

## Loss and Metrics

Both the loss function and metrics used for validation and testing are configured through `MetricsManager` objects (see docs [here](../api/metrics.rst)), as argument to the `training_module` [block in the config](config.md/#training_module).
Details are provided in the `MetricsManager` [docs page](../api/metrics.rst).
Simplified wrappers are provided for typical use cases, i.e. the `EnergyForceLoss` and `EnergyForceStressLoss` loss function wrappers, and the `EnergyForceMetrics` and `EnergyForceStressMetrics` metrics wrappers.