"""Example script to make a parity plot from the results of using `nequip.train.callbacks.TestTimeXYZFileWriter`.

Thanks to Hongyu Yu for useful input: https://github.com/mir-group/nequip/discussions/223#discussioncomment-4923323
"""

import argparse
import numpy as np

import matplotlib.pyplot as plt

import ase.io

# Parse arguments:
parser = argparse.ArgumentParser(
    description="Make a parity plot from the results of using `nequip.train.callbacks.TestTimeXYZFileWriter`."
)
parser.add_argument(
    "xyzoutput",
    help=".xyz file from using `nequip.train.callbacks.TestTimeXYZFileWriter`",
)
parser.add_argument("--output", help="File to write plot to", default=None)
parser.add_argument(
    "--per-atom", action="store_true", help="Normalize energy by number of atoms"
)
args = parser.parse_args()

forces = []
ref_forces = []
energies = []
ref_energies = []
for frame in ase.io.iread(args.xyzoutput):
    forces.append(frame.get_forces().flatten())
    ref_forces.append(frame.arrays["original_dataset_forces"].flatten())
    n_atoms = len(frame) if args.per_atom else 1.0
    energies.append(frame.get_potential_energy() / n_atoms)
    ref_energies.append(frame.info["original_dataset_energy"] / n_atoms)
forces = np.concatenate(forces, axis=0)
ref_forces = np.concatenate(ref_forces, axis=0)
energies = np.asarray(energies)
ref_energies = np.asarray(ref_energies)

# energy metrics
energy_errors = energies - ref_energies
energy_mae = np.mean(np.abs(energy_errors))
energy_rmse = np.sqrt(np.mean(energy_errors**2))
energy_maxae = np.max(np.abs(energy_errors))
energy_header = "Energy (per atom) Metrics:" if args.per_atom else "Energy Metrics:"
print(energy_header)
print(f"  MAE:   {energy_mae:.6f}")
print(f"  RMSE:  {energy_rmse:.6f}")
print(f"  MaxAE: {energy_maxae:.6f}")

# force metrics
force_errors = forces - ref_forces
force_mae = np.mean(np.abs(force_errors))
force_rmse = np.sqrt(np.mean(force_errors**2))
force_maxae = np.max(np.abs(force_errors))
print()
print("Component-Wise Force Metrics:")
print(f"  MAE:   {force_mae:.6f}")
print(f"  RMSE:  {force_rmse:.6f}")
print(f"  MaxAE: {force_maxae:.6f}")

fig, axs = plt.subplots(ncols=2, figsize=(8, 4))

energy_label = "energy per atom" if args.per_atom else "energy"
ax = axs[0]
ax.set_xlabel(f"Reference {energy_label}")
ax.set_ylabel(f"Model {energy_label}")
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="-", color="k", alpha=0.7)
ax.scatter(ref_energies, energies, s=5, color="g")
ax.set_aspect("equal")

ax = axs[1]
ax.set_xlabel("Reference force component")
ax.set_ylabel("Model force component")
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="-", color="k", alpha=0.7)
ax.scatter(ref_forces, forces, s=5, color="g")
ax.set_aspect("equal")

plt.suptitle("Parity Plots")

plt.tight_layout()
if args.output is None:
    plt.show()
else:
    plt.savefig(args.output)
