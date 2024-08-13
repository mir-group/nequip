nequip.data (Fields, Modifiers, and Statistics)
===============================================

Data Fields
###########

The NequIP infrastructure provides some ready-to-use data fields, such as ``total_energy``, ``forces``, ``stress``, etc. These are the names that should be referred to when using methods and classes in the NequIP package, such as the fields given to ``nequip.train.MetricsManager``. The data fields are broadly categorized (in a mutually exclusive manner) as graph (per-frame), node (per-atom), or edge fields (per-"bond"). 

.. autoclass:: nequip.data._GRAPH_FIELDS
.. autoclass:: nequip.data._NODE_FIELDS
.. autoclass:: nequip.data._EDGE_FIELDS

There are additional categories used for the internal data processing in the NequIP infrastructure.

.. autoclass:: nequip.data._LONG_FIELDS

.. autoclass:: nequip.data._CARTESIAN_TENSOR_FIELDS

Custom fields must be registered with the following field registration methods to be compatible with the internal logic of NequIP's data processing infrastructure.

.. autofunction:: nequip.data.register_fields

.. autofunction:: nequip.data.deregister_fields



Data Modifiers
##############

 .. autoclass:: nequip.data.PerAtomModifier
    :members:

 .. autoclass:: nequip.data.EdgeLengths
    :members:

 .. autoclass:: nequip.data.NumNeighbors
    :members:


Dataset Statistics
##################

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