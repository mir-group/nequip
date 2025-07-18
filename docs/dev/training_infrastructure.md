# Training Infrastructure

This page explains the core training architecture built on PyTorch Lightning.

## Training Modules

The core abstraction for managing the training loop is the [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html).
Our base class for this abstraction is {class}`~nequip.train.NequIPLightningModule`.
New training techniques (that require more control of the training loop than what [callbacks](https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks) can offer) should be implemented as subclasses of {class}`~nequip.train.NequIPLightningModule`.
Examples that currently exist in `nequip` include the {class}`~nequip.train.EMALightningModule` (which holds an exponential moving average of the base model's weights), the {class}`~nequip.train.ConFIGLightningModule` (which implements the [Conflict-Free Inverse Gradients](https://arxiv.org/abs/2408.11104) method for multitask training, e.g. to balance energy and force loss gradients), and the {class}`~nequip.train.EMAConFIGLightningModule` (which combines the EMA and ConFIG training strategies).

We aim for modularity of new training techniques, at the cost of proliferating various combinations of techniques.
This design choice is motivated by the fact that not all implemented training techniques are seamlessly composable with one another, and thought has to be put into composing them anyway.

Because of the potential need to compose {class}`~nequip.train.NequIPLightningModule` subclasses, several rules should be obeyed to limit the possibility of silent errors. Note that composing {class}`~nequip.train.NequIPLightningModule` subclasses takes the form of the "deadly diamond of death", a notorious multiple inheritance pattern that developers must be aware and cautious of when writing compositions.

- class attributes specific to a subclass should have a unique, distinguishable name to avoid the possibility of overwriting variables when attempting multiple inheritance (a clean way might be to use a dataclass)
- be very careful of the order of inheritance when creating "diamond" subclasses (a subclass that derives from other subclasses of {class}`~nequip.train.NequIPLightningModule`), and use assertions to make sure that the new training module behaves as intended