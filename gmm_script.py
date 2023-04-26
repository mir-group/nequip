"""Example script to plot GMM uncertainties vs. atomic force errors from the results of `nequip-evaluate`"""

import math
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ase.io import read
from sklearn.metrics import mean_squared_error

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
# will evaluate deployed_model.pth AND the fitted GMM on the data set at
# path/to/training/session and will write the NLLs and the true atomic forces (along
# with the typical outputs of `nequip-evaluate`) to out.xyz. This script can then use
# out.xyz to create a plot of NLL vs. atomic force RMSE.

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

# Read in results of NequIP model evaluation
atoms_list = read(args.xyzoutput, index=":", format="extxyz")

# Number of data points (aspirin molecules) in original data set
num_data_pts = len(atoms_list)

# Number of atoms per data point (aspirin molecule)
num_atoms_per_pt = len(atoms_list[0])

# Extract predicted forces, true forces, and per-atom NLLs from evaluation
for atoms in atoms_list:
    pred_forces.append(atoms.get_forces())
    true_forces.append(atoms.get_array("original_dataset_forces"))
    nlls.append(atoms.get_array("node_features_nll"))
pred_forces = np.asarray(pred_forces)
true_forces = np.asarray(true_forces)
nlls = np.asarray(nlls)

# Compute per-atom RMSE of force predictions
force_rmses = np.zeros((num_data_pts, num_atoms_per_pt))
for mol_idx in range(num_data_pts):
    for atom_idx in range(num_atoms_per_pt):
        force_rmses[mol_idx][atom_idx] = math.sqrt(
            mean_squared_error(
                true_forces[mol_idx][atom_idx],
                pred_forces[mol_idx][atom_idx],
            )
        )

# Plot per-atom NLL vs. per-atom force RMSE
f = plt.figure(figsize=(6, 6))
plt.hist2d(
    force_rmses.flatten(),
    nlls.flatten(),
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
