# Loss and Metrics

Loss functions and metrics are configured by specifying a field (e.g. `total_energy`, `forces`, [etc.](../api/data_fields.rst)) and an error quantity to calculate for it (e.g. {class}`~nequip.train.MeanSquaredError`, {class}`~nequip.train.MeanAbsoluteError`, etc).

Loss functions and metrics are configured through {class}`~nequip.train.MetricsManager` objects in the [`training_module`](config.md/#training_module) section of the config.
The loss function determines what the model optimizes during training, while metrics are used for monitoring training progress and conditioning training behavior (early stopping, learning rate scheduling, etc.).

## Units
All loss components and metrics are in the physical units associated with the dataset.
For example, if the dataset uses force units of eV/Å, a force mean-squared error (MSE) would have units of (eV/Å)².

## Simplified Wrappers

Most users should use the simplified wrapper classes for common force field training scenarios. These wrappers automatically configure the appropriate metrics for you:

**For Loss Functions:**
- {class}`~nequip.train.EnergyForceLoss`
- {class}`~nequip.train.EnergyForceStressLoss`

**For Validation/Test Metrics:**
- {class}`~nequip.train.EnergyForceMetrics`
- {class}`~nequip.train.EnergyForceStressMetrics`

When using simplified wrappers, the actual metric names logged during training may not be immediately obvious. Each wrapper creates specific metrics with predetermined names. To see exactly what metric names each wrapper produces, refer to their individual API documentation in the [`nequip.train` metrics API reference](../api/metrics.rst).

## Coefficients and Weighted Sum

Users can set coefficients (`coeff`) for each loss or metric term, which leads to the computation of a `weighted_sum` metric.

1. For **loss functions**, `weighted_sum` is the actual loss value used for backpropagation.
2. For **validation/test metrics**, `weighted_sum` provides a single monitoring metric that balances multiple quantities to be used for conditioning checkpointing, early stopping, learning rate scheduling, etc.

Coefficients are automatically normalized to sum to 1. For example:
```yaml
coeffs:
  total_energy: 3.0
  forces: 1.0
```
becomes internally: `total_energy: 0.75, forces: 0.25`.

The `weighted_sum` is calculated as:
```
weighted_sum = (coeff_1 * metric_1) + (coeff_2 * metric_2) + ...
```
Coefficients only affect the `weighted_sum` calculation. The individual metrics (e.g., `total_energy_rmse`, `forces_rmse`) are logged with their actual computed values, unmodified by coefficients.

Metrics with `coeff: null` (or omitted from `coeffs`) are still computed and logged, but excluded from `weighted_sum`:

```yaml
coeffs:
  total_energy_rmse: 1.0    # included in weighted_sum
  forces_rmse: 1.0          # included in weighted_sum
  total_energy_mae: null    # computed but not in weighted_sum
  forces_mae: null          # computed but not in weighted_sum
```

Here's an example showing how to set up metrics and use `weighted_sum` for monitoring:

```yaml
# Define the monitored metric once for consistency
monitored_metric: val0_epoch/weighted_sum

training_module:
  _target_: nequip.train.EMALightningModule
  
  # Loss function
  loss:
    _target_: nequip.train.EnergyForceLoss
    coeffs:
      total_energy: 1.0
      forces: 1.0
  
  # Validation metrics - weighted_sum will be used for monitoring
  val_metrics:
    _target_: nequip.train.EnergyForceMetrics
    coeffs:
      total_energy_rmse: 1.0
      forces_rmse: 1.0
      total_energy_mae: null  # logged but not in weighted_sum
      forces_mae: null        # logged but not in weighted_sum

trainer:
  _target_: lightning.Trainer
  
  callbacks:
    # Early stopping using the monitored metric
    - _target_: lightning.pytorch.callbacks.EarlyStopping
      monitor: ${monitored_metric}
      patience: 20
      min_delta: 1e-3
    
    # Model checkpointing using the monitored metric  
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      monitor: ${monitored_metric}
      filename: best

  # Learning rate scheduler using the monitored metric
  lr_scheduler:
    scheduler:
      _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
      factor: 0.6
      patience: 5
    monitor: ${monitored_metric}
```

## Advanced Usage: Custom MetricsManager

For scenarios not covered by the simplified wrappers, you can use the full {class}`~nequip.train.MetricsManager` directly. Technical details and advanced examples are provided in the [`nequip.train.MetricsManager` API documentation](../api/metrics.rst).

Common advanced use cases include:
- Custom field modifiers beyond {class}`~nequip.data.PerAtomModifier`
- Per-type metrics (separate metrics for each atom type)
- Custom metric types (e.g., {class}`~nequip.train.HuberLoss`, {class}`~nequip.train.StratifiedHuberForceLoss`)
- Handling datasets with missing labels (using `ignore_nan: true`)
