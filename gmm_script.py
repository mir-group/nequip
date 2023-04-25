import pathlib
import subprocess
import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ase.io import read
from sklearn.metrics import mean_squared_error

# GMM uncertainties example script using NequIP model trained on minimal.yaml #

# Train NequIP model on minimal.yaml
# Make sure to remove results/aspirin/minimal before running this
retcode = subprocess.run(
    ["nequip-train", "configs/minimal.yaml"],
    cwd=pathlib.Path(__file__).parents[0],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
retcode.check_returncode()
print("Training complete")

# Deploy trained NequIP model
retcode = subprocess.run(
    [
        "nequip-deploy",
        "build",
        "--using-dataset",
        "--model",
        "configs/minimal_gmm.yaml",
        "out.pth",
    ],
    cwd=pathlib.Path(__file__).parents[0],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
retcode.check_returncode()
print("Deployment complete")

# Evaluate deployed NequIP model on original data set
retcode = subprocess.run(
    [
        "nequip-evaluate",
        "--train-dir",
        "results/aspirin/minimal",
        "--model",
        "out.pth",
        "--output",
        "minimal_gmm.xyz",
        "--output-fields",
        "node_features_nll",
        "--output-fields-from-original-dataset",
        "forces",
    ],
    cwd=pathlib.Path(__file__).parents[0],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
retcode.check_returncode()
print("Evaluation complete")

pred_forces = []
true_forces = []
nlls = []

# Read in results of NequIP model evaluation
atoms_list = read("minimal_gmm.xyz", index=":", format="extxyz")

# Number of data points (aspirin molecules) in original data set
num_data_pts = len(atoms_list)

# Number of atoms per data point (aspirin molecule)
num_atoms_per_pt = len(atoms_list[0])

# Extract predicted forces, true forces, and per-atom NLLs from evaluation
for atoms in atoms_list:
    pred_forces.append(atoms.get_forces())
    true_forces.append(atoms.get_array("original_dataset_forces"))
    nlls.append(atoms.get_array("node_features_nll"))
pred_forces = np.array(pred_forces)
true_forces = np.array(true_forces)
nlls = np.array(nlls)

# print(f"pred_forces.shape: {pred_forces.shape}")
# print(f"true_forces.shape: {true_forces.shape}")
# print(f"nlls.shape: {nlls.shape}")

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
plt.title("NLL vs. Atomic Force RMSE (Aspirin)")
plt.xlabel(r"RMSE, $\epsilon$ $(\mathrm{meV/\AA})$")
plt.ylabel(r"Negative Log Likelihood, NLL")
plt.grid(linestyle="--")
plt.savefig("minimal_gmm.png", bbox_inches="tight")
