"""Example script to plot GMM uncertainties vs. atomic force errors from the results of `nequip-evaluate`"""

import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ase.io import read

# To obtain GMM uncertainties for each atom in a system, a NequIP model must be trained
# (e.g., using `nequip-train configs/minimal.yaml`) and then deployed. To fit a GMM
# during deployment, run
#
#   `nequip-deploy build --using-dataset --model deployment.yaml deployed_model.pth`
#
# where deployment.yaml is a config file that adds and fits a GMM to the deployed model
# (for an example, see configs/minimal_gmm.yaml). Lastly, to obtain negative log
# likelihoods (NLLs) on some test data, the NequIP model must be evaluated on a data set using
# `nequip-evaluate` with `--output-fields node_features_nll` and
# `--output-fields-from-original-dataset forces`. For example, running
#
#   `nequip-evaluate --train-dir path/to/training/session --model deployed_model.pth --output out.xyz --output-fields node_features_nll --output-fields-from-original-dataset forces`
#
# will evaluate deployed_model.pth AND the fitted GMM on the data set in the config at
# path/to/training/session and will write the NLLs and the true atomic forces (along
# with the typical outputs of `nequip-evaluate`) to out.xyz. IMPORTANT: The data set
# config must have contain the lines
#   `node_fields:
#      - node_features_nll`
# in order for nequip-evaluate to recognize `node_features_nll` as a legitimate argument.
# This script can then use out.xyz to create a plot of NLL vs. atomic force RMSE.

# Parse arguments
parser = argparse.ArgumentParser(
    description="Make a plot of GMM NLL uncertainty vs. atomic force RMSE from the results of `nequip-evaluate`."
)
parser.add_argument(
    "xyzoutput",
    help=".xyz file from running `nequip-evaluate ... --output out.xyz --output-fields node_features_nll --output-fields-from-original-dataset forces",
)
parser.add_argument("--output", help="File to write plot to", default=None)
args = parser.parse_args()

pred_forces = []
true_forces = []
nlls = []

# Extract predicted forces, true forces, and per-atom NLLs from evaluation
for frame in read(args.xyzoutput, index=":", format="extxyz"):
    pred_forces.append(frame.get_forces())
    true_forces.append(frame.get_array("original_dataset_forces"))
    nlls.append(frame.get_array("node_features_nll"))
pred_forces = np.concatenate(pred_forces, axis=0)
true_forces = np.concatenate(true_forces, axis=0)
nlls = np.concatenate(nlls, axis=0)

# Compute force RMSE for each atom
force_rmses = np.sqrt(np.mean(np.square(true_forces - pred_forces), axis=-1))

# Plot per-atom NLL vs. per-atom force RMSE
f = plt.figure(figsize=(6, 6))
plt.hist2d(
    force_rmses,
    nlls,
    bins=(100, 100),
    cmap="viridis",
    norm=mpl.colors.LogNorm(),
    cmin=1,
)
plt.title("NLL vs. Atomic Force RMSE")
plt.xlabel(r"RMSE, $\epsilon$ $(\mathrm{meV/\AA})$")
plt.ylabel(r"Negative Log Likelihood, NLL")
plt.grid(linestyle="--")
plt.tight_layout()
if args.output is None:
    plt.show()
else:
    plt.savefig(args.output)
