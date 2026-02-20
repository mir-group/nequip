# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import warnings

from nequip.integrations.ase import NequIPCalculator

warnings.warn(
    "DEPRECATION WARNING: `nequip.ase.NequIPCalculator` has moved to "
    "`nequip.integrations.ase.NequIPCalculator`. Please update imports now; "
    "the old path will be removed in a future release.",
    FutureWarning,
    stacklevel=2,
)

__all__ = ["NequIPCalculator"]
