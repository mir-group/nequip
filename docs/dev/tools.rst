Developer Tools
===============

Custom Model Builders
---------------------
The following decorator should be used for new model builders, e.g. ::

  @nequip.model.model_builder
  def my_new_model_builder(arg1, arg2):
      return model(arg1, arg2)


.. autofunction:: nequip.model.model_builder