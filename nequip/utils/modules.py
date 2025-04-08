# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import Optional

import torch


def find_first_of_type(m: torch.nn.Module, kls) -> Optional[torch.nn.Module]:
    """Find the first module of a given type in a module tree."""
    if isinstance(m, kls):
        return m
    else:
        for child in m.children():
            tmp = find_first_of_type(child, kls)
            if tmp is not None:
                return tmp
    return None
