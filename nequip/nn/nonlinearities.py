# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

import math

# Technically, use of this module should probably be guarded by conditional_torchscript_jit
# But its use as a drop-in replacement for functions like torch.nn.functional.silu makes that
# difficult, so, given the rarety of its use, we have just removed @torch.jit.script


def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)
