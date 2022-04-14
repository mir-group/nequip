"""Example of patching a model after training to analyze it.

This file shows how to load a pickled Python model after training
and modify it to save and output the features after the first
convolution for later analysis.
"""

from nequip.utils import Config, find_first_of_type
from nequip.data import AtomicDataDict, AtomicData, dataset_from_config
from nequip.nn import SequentialGraphNetwork, SaveForOutput
from nequip.train import Trainer

# The path to the original training session
path = "../results/aspirin/minimal"

# Load the model
# there are other ways to do this, such as model_from_config etc.
model = Trainer.load_model_from_training_session(traindir=path)

# Find the SequentialGraphNetwork, which contains the
# sequential bulk of the NequIP GNN model. To see the
# structure of the GNN models, see
# nequip/models/_eng.py
sgn = find_first_of_type(model, SequentialGraphNetwork)

# Now insert a SaveForOutput
insert_after = "layer1_convnet"  # change this, again see nequip/models/_eng.py
# `insert_from_parameters` builds the module for us from `shared_parameters`
# You could also build it manually and use `insert`, but `insert_from_parameters`
# has the advantage of constructing it with the correct input irreps
# based on whatever comes before:
sgn.insert_from_parameters(
    after=insert_after,
    name="feature_extractor",
    shared_params=dict(
        field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field="saved",
    ),
    builder=SaveForOutput,
)

# Now, we can test our patched model:
# Load the original config file --- this could be a new one too:
config = Config.from_file(path + "/config.yaml")
# Load the dataset:
# (Note that this loads the training dataset if there are separate training and validation datasets defined.)
dataset = dataset_from_config(config)

# Evaluate the model on a configuration:
data = dataset[0]
out = sgn(AtomicData.to_AtomicDataDict(data))

# Check that our extracted data is there:
assert "saved" in out
print(out["saved"].shape)
