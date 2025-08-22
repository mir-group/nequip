Custom Models
=============

Extension packages implementing custom models should be aware of the following model building infrastructure pieces in NequIP.

Model Builders
--------------

Model builders are functions decorated with :func:`~nequip.model.model_builder` that construct models with proper handling of floating point precision, seeding, and compilation options:

.. code-block:: python

   @nequip.model.model_builder
   def my_new_model_builder(arg1, arg2):
       return model(arg1, arg2)

.. autofunction:: nequip.model.model_builder

Model Modifiers
---------------

Model modifiers are class methods decorated with :func:`~nequip.nn.model_modifier_utils.model_modifier` that can modify loaded models on-the-fly:

.. code-block:: python

   @model_modifier(persistent=True)
   @classmethod
   def modify_PerTypeScaleShift(cls, model, scales=None, shifts=None, ...):
       # Implementation here
       pass

The ``persistent`` parameter determines whether the modifier is applied during model packaging:

- **Non-persistent** (``persistent=False``): Applied only at runtime, often for accelerations
- **Persistent** (``persistent=True``): Applied during packaging, for structural changes

**Important**: Model modifiers MUST preserve model state by transferring weights when there are trainable parameters.

Use :func:`~nequip.nn.model_modifier_utils.replace_submodules` to help implement modifiers that replace specific module types.

**Usage:**

Train-time via config:

.. code-block:: yaml

   model:
     _target_: nequip.model.modify
     modifiers:
       - modifier: modify_PerTypeScaleShift
         shifts:
           C: 1.23
           H: 0.12
     model:
       _target_: nequip.model.ModelFromPackage
       # ... 

Compile-time for accelerations:

.. code-block:: bash

   nequip-compile model.ckpt compiled.pth --modifiers enable_OpenEquivariance

Implementation Tips
-------------------

When implementing custom :class:`torch.nn.Module` subclasses, use :meth:`torch.nn.Module.extra_repr` to provide crucial model information for debugging (visible when ``_NEQUIP_LOG_LEVEL=DEBUG`` is set).