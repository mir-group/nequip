"""Example of loading a NequIP dataset and computing its RDFs"""

import argparse
import itertools

from scipy.special import comb
import matplotlib.pyplot as plt

from nequip.utils import Config
from nequip.data import dataset_from_config
from nequip.scripts.train import default_config
from nequip.utils._global_options import _set_global_options

# Parse arguments:
parser = argparse.ArgumentParser(
    description="Plot RDFs of dataset specified in a `nequip` YAML file"
)
parser.add_argument("config", help="YAML file configuring dataset")
args = parser.parse_args()
config = Config.from_file(args.config, defaults=default_config)
_set_global_options(config)

print("Loading dataset...")
r_max = config["r_max"]
dataset = dataset_from_config(config=config)
print(
    f"    loaded dataset of {len(dataset)} frames with {dataset.type_mapper.num_types} types"
)

print("Computing RDFs...")
rdfs = dataset.rdf(bin_width=0.01)

print("Plotting...")
num_types: int = dataset.type_mapper.num_types
fig, axs = plt.subplots(nrows=int(comb(N=num_types, k=2)), sharex=True)

for i, (type1, type2) in enumerate(itertools.combinations(range(num_types), 2)):
    ax = axs[i]
    ax.set_ylabel(
        f"{dataset.type_mapper.type_names[type1]}-{dataset.type_mapper.type_names[type2]} RDF"
    )
    hist, bin_edges = rdfs[(type1, type2)]
    ax.plot(bin_edges[:-1], hist)

ax.set_xlabel("Distance")

plt.show()
