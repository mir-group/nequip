Under The Hood
##############

This page explains key aspects of the infrastructure.


Training Modules
================

The core abstraction for managing the training loop is the `LightningModule <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`_.
Our base class for this abstraction is ``nequip.train.NequIPLightningModule``.
New training techniques (that require more control of the training loop than what `callbacks <https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks>`_ can offer) should be implemented as subclasses of ``NequIPLightningModule``.
Examples that currently exist in ``nequip`` include the ``EMALightningModule`` (which holds an exponential moving average of the base model's weights), the ``ConFIGLightningModule`` (which implements the `Conflict-Free Inverse Gradients <https://arxiv.org/abs/2408.11104>`_ method for multitask training, e.g. to balance energy and force loss gradients), and the ``EMAConFIGLightningModule`` (which combines the EMA and ConFIG training strategies).

We aim for modularity of new training techniques, at the cost of proliferating various combinations of techniques.
This design choice is motivated by the fact that not all implemented training techniques are seamlessly composable with one another, and thought has to be put into composing them anyway.

Because of the potential need to compose ``NequIPLightningModule`` subclasses, several rules should be obeyed to limit the possibility of silent errors. Note that composing ``NequIPLightningModule`` subclasses takes the form of the "deadly diamond of death", a notorious multiple inheritance pattern that developers must be aware and cautious of when writing compositions.

 - class attributes specific to a subclass should have a unique, distinguishable name to avoid the possibility of overwriting variables when attempting multiple inheritance (a clean way might be to use a dataclass)
 - be very careful of the order of inheritance when creating "diamond" subclasses (a subclass that derives from other subclasses of ``NequIPLightningModule``), and use assertions to make sure that the new training module behaves as intended



Model Building
==============

Model Builder Decorator
-----------------------
The following decorator should be used for new model builders, e.g. ::

  @nequip.model.model_builder
  def my_new_model_builder(arg1, arg2):
      return model(arg1, arg2)


.. autofunction:: nequip.model.model_builder

Dtypes in Model Building
------------------------
``model_dtype`` is imposed by using ``torch.set_default_dtype()`` to set the default ``dtype`` to ``model_dtype`` for the duration of model building, such that parameter tensors created during model building will be in the default ``dtype``, which was set to ``model_dtype``.


Modules
=======
Try to use ``extra_repr`` (see `here <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.extra_repr>`_) when defining custom ``torch.nn.Module`` subclasses to convey crucial information about the model. (The model structure is printed before training if ``_NEQUIP_LOG_LEVEL=DEBUG`` environment variable is set.)

