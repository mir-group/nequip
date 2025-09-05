# FAQs

## Logging

  **Q**: How does logging work? How do I use Tensorboard or Weights and Biases?

  **A**: Logging is configured under the [`trainer`](../configuration/config.md#trainer) section of the config file by specifying the `logger` argument of the {class}`~lightning.pytorch.trainer.trainer.Trainer`. Compatible loggers are found [here](https://lightning.ai/docs/pytorch/stable/api_references.html#loggers). Read the [Config](../configuration/config.md) docs for a more complete description.

## Units
  **Q**: What units do `nequip` framework models use?

  **A**: `nequip` has no preferred system of units and uses the units of the data provided. Model inputs, outputs, error metrics, and all other quantities follow the units of the dataset. Users **must** use consistent input and output units. For example, if the length unit is Å and the energy labels are in eV, the force predictions from the model will be in eV/Å. The provided force labels should hence also be in eV/Å. 

```{warning}
`nequip` cannot and does not check the consistency of units in inputs you provide, and it is your responsibility to ensure consistent treatment of input and output units
```

## Floating Point Precision

  **Q**: What floating point precision (`torch.dtype`) is used in the `nequip` framework?

  **A**: `float64` precision is used for data (inputs to model and reference labels). Either `float32` or `float64` precision can be used as the `model_dtype` (which is a mandatory hyperparameter of models in the `nequip` framework). If `float32` precision is used for `model_dtype`, the model will cast down from the `float64` inputs (e.g. positions) and cast up the outputs (e.g. energy) to `float64`. A major change in the post-revamp `nequip` framework is that NequIP or Allegro models keep the initial embeddings in `float64` before casting down if `model_dtype=float32` for better numerics.

## Validation metrics are much better than training metrics or loss

  **Q**: The same type of metric (e.g. force MAE) is a lot lower on the validation set than the training set during the course of training. What's happening?

  **A**: This phenomenon is generally observed when using {class}`~nequip.train.EMALightningModule` as the training module, where validation (and inference tasks in general) uses an exponential-moving average (EMA) of the weights that vary more rapidly during training. Thus, training and validation happens on a different set of model weights, leading to the differences. The better validation metrics justifies why the EMA approach is useful in practice. The answer would be different if this phenomenon is observed without EMA.

## Distributed Training

  **Q**: How do I train with multiple GPUs?

  **A**: Read our [Distributed Data Parallel training docs](../accelerations/ddp_training.md).

## AMD GPU Compatibility

  **Q**: Does the NequIP framework support AMD GPUs?

  **A**: The NequIP framework is compatible with AMD GPUs. However, certain acceleration features, including [CuEquivariance](../accelerations/cuequivariance.md), require NVIDIA GPUs. When specifying device parameters, use `cuda` as the device identifier for both NVIDIA and AMD GPUs.

## Energy-Only Training

  **Q**: How do I train on datasets that only contain energies (no forces)?

  **A**: For energy-only datasets, use the following specialized components:

  - **Model**: {class}`~nequip.model.NequIPGNNEnergyModel` instead of {class}`~nequip.model.NequIPGNNModel`
  - **Data Statistics**: {class}`~nequip.data.EnergyOnlyDataStatisticsManager` instead of {class}`~nequip.data.CommonDataStatisticsManager`
  - **Loss Function**: {class}`~nequip.train.EnergyOnlyLoss` instead of {class}`~nequip.train.EnergyForceLoss`
  - **Validation Metrics**: {class}`~nequip.train.EnergyOnlyMetrics` instead of {class}`~nequip.train.EnergyForceMetrics`

  These components are specifically designed for energy-only training and will not attempt to compute force-related statistics or metrics that would cause errors with datasets lacking force labels.

## Upgrading from pre-`0.7.0` `nequip`

```{warning}
Importing, restarting, or migrating models or training runs from pre-0.7.0 versions of `nequip` is not supported.  Please use Python environment management to maintain separate installations of older `nequip` versions to keep working with that data, if necessary.
```

  **Q**: What replaces `nequip-evaluate`, which was removed?

  **A**: `nequip-evaluate` is replaced by using the `test` [run type](../configuration/config.md#run) with [`nequip-train`](../getting-started/workflow.md#testing) with the {class}`~nequip.train.callbacks.TestTimeXYZFileWriter` [callback](../../api/callbacks.rst)

  **Q**: What replaces `nequip-deploy`, which was removed?

  **A**: `nequip-deploy` (which previously generates a TorchScript `.pth` file) is replaced by [`nequip-compile`](../getting-started/workflow.md#compilation) that can produce either a TorchScript `.nequip.pth` file or an AOTInductor `.nequip.pt2` file to be used for inference tasks in our [integrations](../../integrations/all.rst) such as ASE and [LAMMPS](../../integrations/lammps/index.md).
  **Q**: What replaces `nequip-benchmark`, which has been removed?

  **A**: No direct substitute exists, but the NequIP ASE calculator can be used to similarly run a model from Python on a single static frame.

  **Q**: Are losses still sometimes in normalized internal units?

  **A**: No, in `nequip >= 0.7.0`, the loss components are all in physical units.