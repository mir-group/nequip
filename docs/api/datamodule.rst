nequip.data (DataModules, Datasets, and Transforms)
===================================================

Data Modules
############

``nequip`` provides a general base ``DataModule`` class, ``NequIPDataModule``,

  * from which subclasses can be built upon, and
  * that is the most configurable data module, able to handle diverse ``Dataset`` objects.
  
An ``ASEDataModule`` is also provided as a general wrapper of the ``NequIPDataModule`` if the datasets are all in the same `ASE <https://wiki.fysik.dtu.dk/ase/ase/io/io.html>`_-readable file formats.

Two additional classes, the ``sGDML_CCSD_DataModule`` and ``NequIP3BPADataModule``, are examples of data modules for specific datasets. Such datasets may support auto-downloading capabilities, pre-defined train-test splits and involve more a more minimal set of arguments.

All data modules should (and would) share the following features

  * a ``seed`` is always required for reproducibility
  * ``xxx_dataloader_kwargs``, which refers to the arguments of `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_. Crucially, this is where one would specify the ``batch_size`` (number of frames per batch), ``num_workers``, and ``shuffle`` as common parameters that should be configured.


 .. autoclass:: nequip.data.datamodule.NequIPDataModule
     :members:

 .. autoclass:: nequip.data.datamodule.ASEDataModule
     :members:

 .. autoclass:: nequip.data.datamodule.sGDML_CCSD_DataModule
     :members:

 .. autoclass:: nequip.data.datamodule.NequIP3BPADataModule
     :members:


Datasets
########

 .. autoclass:: nequip.data.dataset.AtomicDataset
    :members:

 .. autoclass:: nequip.data.dataset.ASEDataset
    :members:

 .. autoclass:: nequip.data.dataset.HDF5Dataset
    :members:

 .. autoclass:: nequip.data.dataset.EMTTestDataset
    :members:

 .. autoclass:: nequip.data.dataset.SubsetByRandomSlice
    :members:

 .. autofunction:: nequip.data.dataset.RandomSplitAndIndexDataset


Transforms
##########

Data transforms convert the raw data from the ``Dataset`` to include information necessary for the model to make predictions and perform training. For example, datasets do not usually come with neighborlists, so the ``NeighborListTransform`` is required to convert raw data that only contains positions and energy (and force) labels to additionally include a neighborlist necessary for the model to make predictions.

 .. autoclass:: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
    :members:

 .. autoclass:: nequip.data.transforms.NeighborListTransform
    :members:

 .. autoclass:: nequip.data.transforms.VirialToStressTransform
    :members:
