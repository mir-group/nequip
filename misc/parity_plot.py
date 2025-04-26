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
args = parser.parse_args()

forces = []
true_forces = []
energies = []
true_energies = []
for frame in ase.io.iread(args.xyzoutput):
    forces.append(frame.get_forces().flatten())
    true_forces.append(frame.arrays["original_dataset_forces"].flatten())
    energies.append(frame.get_potential_energy())
    true_energies.append(frame.info["original_dataset_energy"])
forces = np.concatenate(forces, axis=0)
true_forces = np.concatenate(true_forces, axis=0)
energies = np.asarray(energies)
true_energies = np.asarray(true_energies)

fig, axs = plt.subplots(ncols=2, figsize=(8, 4))

ax = axs[0]
ax.set_xlabel("True force component")
ax.set_ylabel("Model force component")
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", color="gray")
ax.scatter(true_forces, forces)
ax.set_aspect("equal")

ax = axs[1]
ax.set_xlabel("True energy")
ax.set_ylabel("Model energy")
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", color="gray")
ax.scatter(true_energies, energies)
ax.set_aspect("equal")

plt.suptitle("Parity Plots")

plt.tight_layout()
if args.output is None:
    plt.show()
else:
    plt.savefig(args.output)
