"""Plot energies of two-atom dimers from a NequIP model."""

import argparse
import itertools

from scipy.special import comb
import matplotlib.pyplot as plt

import torch

from nequip.data import AtomicDataDict, compute_neighborlist_
import hydra

# Parse arguments:
parser = argparse.ArgumentParser(
    description="Plot energies of two-atom dimers from a NequIP model"
)

parser.add_argument(
    "-ckpt_path",
    help="Path to checkpoint file",
    type=str,
)

parser.add_argument(
    "--device", help="Device", default="cuda" if torch.cuda.is_available() else "cpu"
)
parser.add_argument("--output", help="File to write plot to", default=None)
parser.add_argument("--r-min", default=1.0, type=float)
parser.add_argument("--r-max", default=None, type=float)
parser.add_argument("--n-samples", default=500, type=int)
args = parser.parse_args()

print("Loading model... ")

# === get model hyperparameters ===
checkpoint = torch.load(
    args.ckpt_path,
    map_location="cpu",
    weights_only=False,
)

training_module = checkpoint["hyper_parameters"]["info_dict"]["training_module"]
model_cfg = training_module["model"]
model_r_max = float(model_cfg["r_max"])
type_names = model_cfg["type_names"]
num_types = len(type_names)

# === load model ===
training_module = hydra.utils.get_class(training_module["_target_"])
lightning_module = training_module.load_from_checkpoint(args.ckpt_path)
model = lightning_module.model
model.to(args.device)


print("Computing dimers...")
potential = {}
N_sample = args.n_samples
type_combos = [
    list(e) for e in itertools.combinations_with_replacement(range(num_types), 2)
]
N_combos = len(type_combos)
r = torch.zeros(N_sample * N_combos, 2, 3)

if args.r_max is not None:
    max_range = args.r_max
else:
    max_range = model_r_max

rs_one = torch.linspace(args.r_min, max_range, N_sample)
rs = rs_one.repeat([N_combos])
assert rs.shape == (N_combos * N_sample,)
r[:, 1, 0] += rs  # offset second atom along x axis
types = torch.as_tensor(type_combos)
types = types.reshape(N_combos, 1, 2).expand(N_combos, N_sample, 2).reshape(-1)
r = r.reshape(-1, 3)
assert types.shape == r.shape[:1]
N_at_total = N_sample * N_combos * 2
assert len(types) == N_at_total

data = {
    AtomicDataDict.POSITIONS_KEY: r,
    AtomicDataDict.ATOM_TYPE_KEY: types,
}

data[AtomicDataDict.BATCH_KEY] = torch.repeat_interleave(
    torch.arange(r.shape[0] // 2, dtype=torch.long), 2
)
data[AtomicDataDict.NUM_NODES_KEY] = torch.full((r.shape[0] // 2,), 2, dtype=torch.long)
data[AtomicDataDict.PBC_KEY] = torch.full((r.shape[0] // 2, 3), False, dtype=torch.bool)

result = model(
    AtomicDataDict.to_(
        compute_neighborlist_(AtomicDataDict.from_dict(data), r_max=model_r_max),
        args.device,
    )
)

print("Plotting...")
energies = (
    result[AtomicDataDict.TOTAL_ENERGY_KEY]
    .reshape(N_combos, N_sample)
    .cpu()
    .detach()
    .numpy()
)
del result
rs_one = rs_one.cpu().numpy()
nrows = int(comb(N=num_types, k=2, repetition=True))
fig, axs = plt.subplots(
    nrows=nrows,
    ncols=1,
    sharex=True,
    figsize=(6, 2 * nrows),
    dpi=120,
)

for i, (type1, type2) in enumerate(type_combos):
    if nrows == 1:
        ax = axs
    else:
        ax = axs[i]
    ax.set_ylabel(f"{type_names[type1]}-{type_names[type2]}")
    ax.plot(rs_one, energies[i])

ax.set_xlabel("Distance")
plt.suptitle("$E_\\mathrm{total}$ for two-atom pairs")
plt.tight_layout()
if args.output is None:
    plt.show()
else:
    plt.savefig(args.output)
