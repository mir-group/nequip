.. _migration_note:
   
How to migrate to newer NequIP versions
=======================================

(Written for migration from 0.3.3 to 0.4. Nov. 3. 2021)

If the model are mostly the same and there is only some internal
variable changes, it is possible to migrate your NequIP model from the
older version to the newer version.

Upgrade NequIP
--------------

1. Record the old version
~~~~~~~~~~~~~~~~~~~~~~~~~

Go to the code folder in your virtual environment, find out the last
commit that you are using

.. code:: bash

   # bash code
   NEQUIP_FOLDER=$(python -c "import nequip; print(\"/\".join(nequip.__file__.split(\"/\")[:-1]))")
   cd ${NEQUIP_FOLDER}
   pwd
   git show --oneline -s
   OLD_COMMIT=$(git show --oneline -s|awk '{print $1}')

2. Update your main nequip repo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # bash code
   git pull origin main
   pip install -e ./

Obtain the state_dict from the old ver
--------------------------------------

For version before 0.3.3, the ``last_model.pth`` stores the whole pickle
model. So you need to save the ``state_dict()``; otherwise, skip this
section.

1. Back up the old version
~~~~~~~~~~~~~~~~~~~~~~~~~~

Git clone the old commit to a new folder

.. code:: bash

   # bash code
   git clone git@github.com:mir-group/nequip.git -n old_nequip
   cd old_nequip
   git checkout ${OLD_COMMIT}

2. Save the state_dict from the old verion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Go to the old_nequip folder, make sure that your current nequip is
overloaded by local nequip folder. The result of the code below should
show the old_nequip folder instead of the one usually used in the
virtualenv.

.. code:: python

   #python
   import nequip
   print(nequip.__file__)

Load the old model with the old verion in python.

.. code:: python

   # save_state_dict.py
   import torch
   import sys
   model_folder = sys.argv[1]
   old_model=torch.load(
       f"{model_folder}/last_model.pth",
       map_location=torch.device('cpu') # if it operates on CPU
       )
   torch.save(old_model.state_dict(), f"{model_folder}/new_last_model.pth")

Load the state_dict in the new version
--------------------------------------

Go to any other directorys that are not in the old version nequip
folder.

Double check now the ``nequip.__file__`` should locate at the
``${NEQUIP_FOLDER}``

Then try to load the old ``state_dict()`` to the new model.

.. code:: python

   # in new nequip
   import torch
   from nequip.utils import Config
   from nequip.model import model_from_config

   config = Config.from_file("config_final.yaml")

   # only needed for version 0.3.3
   config["train_on_keys"]=["forces", "total_energy"] 
   config["model_builders"] = ["EnergyModel", "PerSpeciesRescale", "ForceOutput", "RescaleEnergyEtc"]

   model = model_from_config(config, initialize=False)

   d = torch.load("new.pth")
   # load the state dict to the new model
   model.load_state_dict(d)

The code will likely to fail. Render some outputs like below:

.. code:: bash

   RuntimeError: Error(s) in loading state_dict for RescaleOutput:
           Missing key(s) in state_dict: "model.func.per_species_rescale.shifts", "model.func.per_species_rescale.scales". 
           Unexpected key(s) in state_dict: "model.func.per_species_scale_shift.shifts", "model.func.per_species_scale_shift.scales", "model.func.radial_basis.cutoff.p", "model.func.radial_basis.cutoff.r_max"

According to this output and the CHANGELOG.md file, we can revise the
dictionary by renaming or removing variables.

.. code:: python

   # rename all parameters listed in the change log as changed.
   d["model.func.per_species_rescale.shifts"]=d.pop("model.func.per_species_scale_shift.shifts")
   d["model.func.per_species_rescale.scales"]=d.pop("model.func.per_species_scale_shift.scales")
   d.pop("model.func.radial_basis.cutoff.p")
   d.pop("model.func.radial_basis.cutoff.r_max")

   # load the state dict to the new model
   model.load_state_dict(d)

   # save the new state dict
   import nequip
   torch.save(model.state_dict(), f"new_last_model_{nequip.__version__}.pth')

Validate the result using nequip-evaluate
-----------------------------------------

Old model
~~~~~~~~~

.. code:: bash

   python nequip/script/evaluate.py 

New model
~~~~~~~~~

.. code:: bash

   nequip-evaluate --train-dir new_model/ --dataset-config data.yaml --output new.xyz

.. code:: yaml

   root: ./
   r_max: 4
   validation_dataset: ase
   validation_dataset_file_name: validate.xyz
   chemical_symbol_to_type:
     H: 0
     C: 1
     O: 2
