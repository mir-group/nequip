# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from typing import List, Final

NEQUIP_AOTI_INPUTS_KEY: Final[str] = "nequip_aoti_inputs"
NEQUIP_AOTI_OUTPUTS_KEY: Final[str] = "nequip_aoti_outputs"


def serialize_aoti_keys(keys: List[str]) -> str:
    assert all(" " not in key for key in keys), (
        f"AOTI field names cannot contain spaces: {keys}"
    )
    return " ".join(keys)


def parse_aoti_keys(serialized_keys: str) -> List[str]:
    return serialized_keys.split()
