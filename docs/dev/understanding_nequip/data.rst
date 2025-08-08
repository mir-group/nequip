Data Key Registration
=====================

NequIP requires all data fields to be registered by type for proper processing.

Field Types
-----------

- **Graph fields**: per-frame (e.g., ``total_energy``, ``stress``)
- **Node fields**: per-atom (e.g., ``forces``, ``positions``) 
- **Edge fields**: per-edge (e.g., ``edge_vectors``)
- **Long fields**: integer dtype (e.g., ``atomic_numbers``)
- **Cartesian tensors**: physical tensors (e.g., ``stress``)

Registration
------------

.. code-block:: python

   from nequip.data import register_fields

   register_fields(
       graph_fields=["custom_energy"],
       node_fields=["custom_forces"], 
       edge_fields=["custom_edge_attr"],
       long_fields=["custom_indices"],
       cartesian_tensor_fields={"custom_tensor": "ij=ji"}
   )

Usage
-----

Register custom fields before using them in:

- Dataset loading
- Loss functions (``MetricsManager``)
- Model outputs

Built-in fields are pre-registered. See API docs for the complete list.

API Reference
-------------

.. autofunction:: nequip.data.register_fields

.. autofunction:: nequip.data.deregister_fields