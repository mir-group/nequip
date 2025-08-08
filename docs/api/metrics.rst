Loss Function and Error Metrics
###############################

For practical usage and configuration guidance, see the :doc:`../guide/configuration/metrics` guide.
This page provides technical API details.

Simplified Wrappers
===================

The following :class:`~nequip.train.MetricsManager` wrappers can be used for common force field training scenarios:

.. autoclass:: nequip.train.EnergyForceLoss
   :members:

.. autoclass:: nequip.train.EnergyForceStressLoss
   :members:

.. autoclass:: nequip.train.EnergyForceMetrics
   :members:

.. autoclass:: nequip.train.EnergyForceStressMetrics
   :members:

The following can be used for energy-only datasets (without forces):

.. autoclass:: nequip.train.EnergyOnlyLoss
   :members:

.. autoclass:: nequip.train.EnergyOnlyMetrics
   :members:

Advanced Configuration: MetricsManager
=======================================

For users who need custom configurations beyond the simplified wrappers, the full :class:`~nequip.train.MetricsManager` API is available.

.. autoclass:: nequip.train.MetricsManager
   :members:



Error Metrics
=============

.. autoclass:: nequip.train.MeanSquaredError
   :members:

.. autoclass:: nequip.train.RootMeanSquaredError
   :members:

.. autoclass:: nequip.train.MeanAbsoluteError
   :members:

.. autoclass:: nequip.train.MaximumAbsoluteError
   :members:

.. autoclass:: nequip.train.HuberLoss
   :members:

.. autoclass:: nequip.train.StratifiedHuberForceLoss
    :members:
