nequip.data.datamodule
######################

``nequip`` provides a general base ``DataModule`` class, :class:`~nequip.data.datamodule.NequIPDataModule`,

  * from which subclasses can be built upon, and
  * that is the most configurable data module, able to handle diverse ``Dataset`` objects.
  
An :class:`~nequip.data.datamodule.ASEDataModule` is also provided as a general wrapper of the :class:`~nequip.data.datamodule.NequIPDataModule` if the datasets are all in the same `ASE <https://wiki.fysik.dtu.dk/ase/ase/io/io.html>`_-readable file formats.

The :class:`~nequip.data.datamodule.sGDML_CCSD_DataModule` and :class:`~nequip.data.datamodule.NequIP3BPADataModule` are examples of data modules for specific datasets.
Such datasets may support auto-downloading capabilities, pre-defined train-test splits and involve a more minimal set of arguments.

All data modules should (and would) share the following features

  * a ``seed`` is always required for reproducibility
  * ``xxx_dataloader``, which refers to a PyTorch :class:`~torch.utils.data.DataLoader`. Crucially, this is where one would specify the ``batch_size`` (number of frames per batch), ``num_workers``, and ``shuffle`` as common parameters that should be configured.


 .. autoclass:: nequip.data.datamodule.NequIPDataModule
     :members:

 .. autoclass:: nequip.data.datamodule.ASEDataModule
     :members:

 .. autoclass:: nequip.data.datamodule.sGDML_CCSD_DataModule
     :members:

 .. autoclass:: nequip.data.datamodule.rMD17DataModule
     :members:

 .. autoclass:: nequip.data.datamodule.MD22DataModule
     :members:

 .. autoclass:: nequip.data.datamodule.NequIP3BPADataModule
     :members:

 .. autoclass:: nequip.data.datamodule.TM23DataModule
     :members: