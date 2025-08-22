Data Handling
=============

Extension packages can implement custom data handling by subclassing NequIP's base classes and registering custom data fields.

Data Key Registration
---------------------

NequIP requires all data fields to be registered by type for proper processing.

Field Types
~~~~~~~~~~~

- **Graph fields**: per-frame (e.g., ``total_energy``, ``stress``)
- **Node fields**: per-atom (e.g., ``forces``, ``positions``) 
- **Edge fields**: per-edge (e.g., ``edge_vectors``)
- **Long fields**: integer dtype (e.g., ``atomic_numbers``)
- **Cartesian tensors**: physical tensors (e.g., ``stress``)

Registration Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nequip.data import register_fields

   register_fields(
       graph_fields=["custom_energy"],
       node_fields=["custom_forces"], 
       edge_fields=["custom_edge_attr"],
       long_fields=["custom_indices"],
       cartesian_tensor_fields={"custom_tensor": "ij=ji"}
   )

Register custom fields in your package's ``__init__.py`` before they are used in:

- Dataset loading
- Loss functions (``MetricsManager``)
- Model outputs

Built-in fields are pre-registered. See the API reference below for the complete details.

API Reference
~~~~~~~~~~~~~

.. autofunction:: nequip.data.register_fields

.. autofunction:: nequip.data.deregister_fields

Custom Datasets
----------------

Extension packages can implement custom datasets by subclassing NequIP's base dataset classes to handle custom data formats and sources.

See :class:`~nequip.data.dataset.AtomicDataset` for the base dataset class and the :doc:`dataset API documentation <../../api/dataset>` for examples.

Custom DataModules
------------------

Extension packages can create custom DataModules to handle specific benchmark datasets or complex data workflows. DataModules manage train/val/test splits, dataset downloading/preprocessing, and coordinate datasets, transforms, dataloaders, and statistics.

See :class:`~nequip.data.datamodule.NequIPDataModule` for the base datamodule class and the :doc:`datamodule API documentation <../../api/datamodule>` for examples of dataset-specific implementations.

Data Transforms
---------------

Extension packages can implement custom data transforms to preprocess data during loading. Transforms are classes that implement a ``__call__`` method to modify ``AtomicDataDict`` objects.

See the :doc:`transforms API documentation <../../api/data_transforms>` for available transform classes and their patterns.