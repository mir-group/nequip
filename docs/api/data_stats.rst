Dataset Statistics
##################

The following :class:`~nequip.data.CommonDataStatisticsManager` can generally be used for common force field training scenarios.

.. autofunction:: nequip.data.CommonDataStatisticsManager

The following can be used for energy-only datasets (without forces):

.. autofunction:: nequip.data.EnergyOnlyDataStatisticsManager

For users who seek to configure their own custom dataset statistics, the following API is offered.

As an example, we show how one can configure the full :class:`~nequip.data.DataStatisticsManager` to have behavior equivalent to using
:class:`~nequip.data.CommonDataStatisticsManager` as follows:

.. code-block:: yaml

   stats_manager:
     _target_: nequip.data.DataStatisticsManager
     type_names: ${model_type_names}
     metrics:
       - name: num_neighbors_mean
         field:
           _target_: nequip.data.NumNeighbors
         metric: 
           _target_: nequip.data.Mean
       - name: per_atom_energy_mean
         field:
           _target_: nequip.data.PerAtomModifier
           field: total_energy
         metric:
           _target_: nequip.data.Mean
       - name: forces_rms
         field: forces
         metric:
           _target_: nequip.data.RootMeanSquare
       - name: per_type_forces_rms
         per_type: true
         field: forces
         metric:
           _target_: nequip.data.RootMeanSquare

.. autoclass:: nequip.data.DataStatisticsManager
    :members:

.. autoclass:: nequip.data.Mean
    :members:

.. autoclass:: nequip.data.MeanAbsolute
    :members:

.. autoclass:: nequip.data.RootMeanSquare
    :members:

.. autoclass:: nequip.data.StandardDeviation
    :members:
    
.. autoclass:: nequip.data.Min
    :members:
    
.. autoclass:: nequip.data.Max
    :members:

.. autoclass:: nequip.data.Count
    :members: