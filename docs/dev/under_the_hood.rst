Under The Hood
##############

This page explains key aspects of the infrastructure.

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