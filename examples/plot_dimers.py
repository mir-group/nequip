"""Plot energies of two-atom dimers from a NequIP model."""

import argparse
import itertools
from pathlib import Path

from scipy.special import comb
import matplotlib.pyplot as plt

import torch

from nequip.data import AtomicData, AtomicDataDict
from nequip.scripts.evaluate import _load_deployed_or_traindir

# Parse arguments:
parser = argparse.ArgumentParser(
    description="Plot energies of two-atom dimers from a NequIP model"
)
parser.add_argument("model", help="Training dir or deployed model", type=Path)
parser.add_argument(
    "--device", help="Device", default="cuda" if torch.cuda.is_available() else "cpu"
)
parser.add_argument("--output", help="File to write plot to", default=None)
parser.add_argument("--r-min", default=1.0, type=float)
parser.add_argument("--r-max", default=None, type=float)
parser.add_argument("--n-samples", default=500, type=int)
args = parser.parse_args()

print("Loading model... ")
model, loaded_deployed_model, model_r_max, type_names = _load_deployed_or_traindir(
    args.model, device=args.device
)
print(f"    loaded{' deployed' if loaded_deployed_model else ''} model")
num_types = len(type_names)

if args.r_max is not None:
    model_r_max = args.r_max

print("Computing dimers...")
potential = {}
N_sample = args.n_samples
N_combs = len(list(itertools.combinations_with_replacement(range(num_types), 2)))
r = torch.zeros(N_sample * N_combs, 2, 3, device=args.device)
rs_one = torch.linspace(args.r_min, model_r_max, N_sample, device=args.device)
rs = rs_one.repeat([N_combs])
assert rs.shape == (N_combs * N_sample,)
r[:, 1, 0] += rs  # offset second atom along x axis
types = torch.as_tensor(
    [list(e) for e in itertools.combinations_with_replacement(range(num_types), 2)]
)
types = types.reshape(N_combs, 1, 2).expand(N_combs, N_sample, 2).reshape(-1)
r = r.reshape(-1, 3)
assert types.shape == r.shape[:1]
N_at_total = N_sample * N_combs * 2
assert len(types) == N_at_total
edge_index = torch.vstack(
    (
        torch.arange(N_at_total, device=args.device, dtype=torch.long),
        torch.arange(1, N_at_total + 1, device=args.device, dtype=torch.long)
        % N_at_total,
    )
)
data = AtomicData(pos=r, atom_types=types, edge_index=edge_index)
data.batch = torch.arange(N_sample * N_combs, device=args.device).repeat_interleave(2)
data.ptr = torch.arange(0, 2 * N_sample * N_combs + 1, 2, device=args.device)
result = model(AtomicData.to_AtomicDataDict(data.to(device=args.device)))

print("Plotting...")
energies = (
    result[AtomicDataDict.TOTAL_ENERGY_KEY]
    .reshape(N_combs, N_sample)
    .cpu()
    .detach()
    .numpy()
)
del result
rs_one = rs_one.cpu().numpy()
nrows = int(comb(N=num_types, k=2, repetition=True))
fig, axs = plt.subplots(
    nrows=nrows,
    sharex=True,
    figsize=(6, 2 * nrows),
    dpi=120,
)

for i, (type1, type2) in enumerate(
    itertools.combinations_with_replacement(range(num_types), 2)
):
    try:
        ax = axs[i]
    except:
        ax = axs
    ax.set_ylabel(f"{type_names[type1]}-{type_names[type2]}")
    ax.plot(rs_one, energies[i])

ax.set_xlabel("Distance")
plt.suptitle("$E_\\mathrm{total}$ for two-atom pairs")
plt.tight_layout()
if args.output is None:
    plt.show()
else:
    plt.savefig(args.output)
