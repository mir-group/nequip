nequip.data.transforms
######################

Data transforms convert the raw data from the ``Dataset`` to include information necessary for the model to make predictions and perform training. For example, datasets do not usually come with neighborlists, so the ``NeighborListTransform`` is required to convert raw data that only contains positions and energy (and force) labels to additionally include a neighborlist necessary for the model to make predictions.

 .. autoclass:: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
    :members:

 .. autoclass:: nequip.data.transforms.NeighborListTransform
    :members:

 .. autoclass:: nequip.data.transforms.VirialToStressTransform
    :members:
