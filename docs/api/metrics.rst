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

Advanced Configuration: MetricsManager
=======================================

For users who need custom configurations beyond the simplified wrappers, the full :class:`~nequip.train.MetricsManager` API is available.

Example: Custom MetricsManager equivalent to EnergyForceLoss:

.. code-block:: yaml

    _target_: nequip.train.MetricsManager
    metrics:
      - name: per_atom_energy_mse
        field:
          _target_: nequip.data.PerAtomModifier
          field: total_energy
        coeff: 1
        metric:
          _target_: nequip.train.MeanSquaredError
      - name: forces_mse
        field: forces
        coeff: 1
        metric:
          _target_: nequip.train.MeanSquaredError

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

.. autoclass:: nequip.train.MeanCubicError
   :members:

.. autoclass:: nequip.train.MeanQuarticError
   :members:

.. autoclass:: nequip.train.HuberLoss
   :members:

.. autoclass:: nequip.train.StratifiedHuberForceLoss
    :members:
