"""Example of patching a model after training to analyze it.

This file shows how to load a pickled Python model after training
and modify it to save and output the features after the first
convolution for later analysis.
"""

from typing import Optional

import torch

from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData
from nequip.nn import SequentialGraphNetwork, SaveForOutput

# The path to the original training session
path = "../results/aspirin/minimal"

# Load the model
model = torch.load(path + "/best_model.pth")


# Define helper function:
def find_first_of_type(m: torch.nn.Module, kls) -> Optional[torch.nn.Module]:
    """Find the first module of a given type in a module tree."""
    if isinstance(m, kls):
        return m
    else:
        for child in m.children():
            tmp = find_first_of_type(child, kls)
            if tmp is not None:
                return tmp
    return None


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
config = Config.from_file(path + "/config_final.yaml")
# Load the dataset:
dataset = dataset_from_config(config)

# Evaluate the model on a configuration:
data = dataset.get(0)
out = sgn(AtomicData.to_AtomicDataDict(data))

# Check that our extracted data is there:
assert "saved" in out
print(out["saved"].shape)
