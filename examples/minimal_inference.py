"""Minimal example of inference with a model in the `nequip` framework.

Author: Albert Musaelian
"""

import torch
import ase.io
from nequip.data import AtomicData, AtomicDataDict
from nequip.scripts.deploy import load_deployed_model

deployed_model = "path_to_deployed_model.pth"
device = "xpu"  # "cpu"
if device == "xpu":
    # Note that we usually use nequip.scripts.deploy.load_deployed_model
    # but that sets some global options (such as TensorFloaat32, which CUDA JIT fuser,
    # etc.) that don't make sense for XPU. So instead here we just the global dtype
    # to `torch.float64`, which is the default in `nequip>=0.6`.
    torch.set_default_dtype(torch.float64)
    # In practice for later production work we should adapt `nequip`'s
    # global configuration infrastructure to correctly handle Intel options.
    model = torch.jit.load(deployed_model)
    model = model.to(device)
    model = model.eval()
    model = torch.jit.freeze(model)
else:
    model, metadata = load_deployed_model(
        deployed_model,
        device=device,
        freeze=True,
        set_global_options=True,  # silence warnings that torch defaults are being overridden
    )

# Load some input structure from an XYZ or other ASE readable file:
data = AtomicData.from_ase(ase.io.read("example_input_structure.xyz"), r_max=4.0)
data = data.to(device)

out = model(AtomicData.to_AtomicDataDict(data))

print(f"Total energy: {out[AtomicDataDict.TOTAL_ENERGY_KEY]}")
print(f"Force on atom 0: {out[AtomicDataDict.FORCE_KEY][0]}")
