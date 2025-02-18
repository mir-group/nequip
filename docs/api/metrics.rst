Loss Function and Error Metrics
###############################

The following API is used to configure loss functions and training/validation/test metrics. One can use ``nequip.train.MetricsManager`` for utmost configurability and flexbility in defining custom loss functions or metrics. For common energy-force and energy-force-stress training scenarios, simplified wrappers are also provided for user convenience.


MetricsManager
==============

 .. autoclass:: nequip.train.MetricsManager
    :members:

MetricsManager Wrappers
=======================

 .. autoclass:: nequip.train.EnergyForceLoss
    :members:

 .. autoclass:: nequip.train.EnergyForceMetrics
    :members:

 .. autoclass:: nequip.train.EnergyForceStressLoss
    :members:

 .. autoclass:: nequip.train.EnergyForceStressMetrics
    :members:


Error Metrics
=============

 .. autoclass:: nequip.train.MeanSquaredError
    :members:

 .. autoclass:: nequip.train.RootMeanSquaredError
    :members:

 .. autoclass:: nequip.train.MeanAbsoluteError
    :members:

 .. autoclass:: nequip.train.HuberLoss
    :members: