Loss Function and Error Metrics
###############################

The following ``MetricsManager`` wrappers can be used for common force field training scenarios, where one seeks to include energyies and forces or energy, forces, and stresses in the loss function or as metrics for monitoring.

 .. autoclass:: nequip.train.EnergyForceLoss
    :members:

 .. autoclass:: nequip.train.EnergyForceStressLoss
    :members:

 .. autoclass:: nequip.train.EnergyForceMetrics
    :members:

 .. autoclass:: nequip.train.EnergyForceStressMetrics
    :members:


For users who seek to configure their own custom loss function or metrics, the following API is offered.

As an example, we show how one can configure the full ``nequip.train.MetricsManager`` to have behavior equivalent to using ``nequip.train.EnergyForceLoss`` as follows::

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

 .. autoclass:: nequip.train.HuberLoss
    :members:

.. autoclass:: nequip.train.StratifiedHuberForceLoss
    :members:
