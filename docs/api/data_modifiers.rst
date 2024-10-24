Data Modifiers
##############

One can use modifiers to convert raw quantities from an ``AtomicDataDict`` into a form that is desired in the ``MetricsManager`` and ``DataStatisticsManager``. For example, 

 - to extract a ``total_energy`` and make it a per-atom ``total_energy`` (``nequip.data.PerAtomModifier``), 
 - to extract and convert position and neighborlist information into edge lengths (``nequip.data.EdgeLengths``), or 
 - to extract and convert neighborlist information into the number of neighbors around each atom (``nequip.data.NumNeighbors``).


 .. autoclass:: nequip.data.PerAtomModifier
    :members:

 .. autoclass:: nequip.data.EdgeLengths
    :members:

 .. autoclass:: nequip.data.NumNeighbors
    :members:
