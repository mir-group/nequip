Training Techniques
===================

Extension packages can implement custom training techniques through two main approaches.

Callbacks
---------

Use PyTorch Lightning `callbacks <https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks>`_ for smaller training customizations that don't alter the core training loop.

As an example, loss coefficients control how different loss terms (energy, forces, stress) are weighted during training - you might want to emphasize forces more heavily in later training epochs. Example callbacks for this featurej include :class:`~nequip.train.callbacks.LossCoefficientScheduler` and :class:`~nequip.train.callbacks.LinearLossCoefficientScheduler`.

Training Modules
----------------

Training modules refer to PyTorch Lightning's `LightningModule <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`_. Subclass :class:`~nequip.train.NequIPLightningModule` for training techniques that require control over weight storage, updates, and manipulation.

Examples include :class:`~nequip.train.EMALightningModule` (exponential moving average), :class:`~nequip.train.ConFIGLightningModule` (multitask training), :class:`~nequip.train.ScheduleFreeLightningModule` (Facebook's Schedule-Free optimizers), and composed modules like :class:`~nequip.train.EMAConFIGLightningModule`.

**Composition Guidelines**

When composing multiple LightningModule subclasses, follow these rules to avoid the "deadly diamond of death" multiple inheritance pattern:

- Use unique, distinguishable names for class attributes to prevent overwrites during inheritance
- Be very careful of inheritance order when creating "diamond" subclasses 
- Use assertions to verify the composed module behaves as intended