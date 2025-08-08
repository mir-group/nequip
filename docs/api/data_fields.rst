Data Fields
###########

The NequIP infrastructure provides some ready-to-use data fields, such as ``total_energy``, ``forces``, ``stress``, etc. These are the names that should be referred to when using methods and classes in the NequIP package, such as the fields given to ``nequip.data.DataStatisticsManager`` or ``nequip.train.MetricsManager``. The data fields are broadly categorized (in a mutually exclusive manner) as graph (per-frame), node (per-atom), or edge fields (per-"bond"). 

.. autodata:: nequip.data._GRAPH_FIELDS
.. autodata:: nequip.data._NODE_FIELDS
.. autodata:: nequip.data._EDGE_FIELDS

There are additional categories used for the internal data processing in the NequIP infrastructure.

.. autodata:: nequip.data._LONG_FIELDS

.. autodata:: nequip.data._CARTESIAN_TENSOR_FIELDS

Custom fields must be registered to be compatible with the internal logic of NequIP's data processing infrastructure. See the :doc:`developer documentation <../dev/understanding_nequip/data>` for registration details.
