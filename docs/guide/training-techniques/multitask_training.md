# Multitask Training

NequIP framework models are typically trained to predict multiple targets simultaneously, such as energies and forces, and sometimes additional properties like stresses or custom targets for specialized applications.

The standard approach to multitask training is to combine multiple loss components using a weighted sum.
Each target (energy, forces, stress, etc.) contributes to the total loss with a user-defined coefficient that controls its relative importance during training.
For detailed information on configuring loss functions and coefficients, see the [Loss Functions and Metrics](../configuration/metrics.md) guide, particularly the sections on [simplified wrapper classes](../configuration/metrics.md#simplified-wrappers) and [coefficient configuration](../configuration/metrics.md#coefficients-and-weighted-sum).

## Advanced Multitask Training Strategies

NequIP provides several advanced techniques organized into training modules and callbacks.

### Training Modules

Training modules are configured in the [`training_module` section](../configuration/config.md#training_module) of your config file:

- **{class}`~nequip.train.ConFIGLightningModule`** - Implements the Conflict-free inverse gradient (ConFIG) approach to multitask learning, which optimizes gradient conflicts between different tasks by solving a linear system to find optimal update directions. See [ConFIG paper](https://arxiv.org/abs/2408.11104).

- **{class}`~nequip.train.EMAConFIGLightningModule`** - Combines the ConFIG approach with exponential moving averages for enhanced stability in multitask scenarios.

### Callbacks

Callbacks are configured in the [`trainer` section](../configuration/config.md#trainer) of your config file and provide dynamic behavior during training:

- **{class}`~nequip.train.callbacks.LossCoefficientScheduler`** - A callback that dynamically adjusts loss coefficients during training based on predefined schedules, allowing you to emphasize different targets at different stages of training.

- **{class}`~nequip.train.callbacks.LossCoefficientMonitor`** - A callback for tracking and logging loss coefficients over time, useful for monitoring how coefficient scheduling affects training dynamics.

- **{class}`~nequip.train.callbacks.SoftAdapt`** - An adaptive callback that automatically adjusts loss coefficients based on the relative rate of learning of different tasks. See [SoftAdapt paper](https://www.sciencedirect.com/science/article/pii/S0927025624003768).
