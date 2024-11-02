import math
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from nequip.data import AtomicDataDict

import difflib
import os
from typing import List


def nequip_make_fx(
    model: torch.nn.Module,
    data: AtomicDataDict.Type,
    fields: List[str],
    extra_inputs: List[torch.Tensor] = [],
    seed: int = 1,
):
    """
    Args:
        model (torch.nn.Module): model must only take in flat ``torch.Tensor`` inputs
        data (AtomicDataDict.Type): an ``AtomicDataDict``
        fields (List[str]): ``AtomicDataDict`` fields that are used as the flat inputs to model
        extra_inputs (List[torch.Tensor]): list of additional ``torch.Tensor`` input data that are not ``AtomicDataDict`` fields
        seed (int): optional seed for reproducibility
    """
    # we do it three times
    # 1. once with the original input data
    test_data_list = [data[key] for key in fields]
    _ = _nequip_make_fx(model, test_data_list + extra_inputs)

    # 2. once with augmented data
    device = data[AtomicDataDict.POSITIONS_KEY].device
    # get global seed to construct generator for reproducibility
    generator = torch.Generator(device).manual_seed(seed)
    num_nodes = AtomicDataDict.num_nodes(data)
    node_idx = torch.randint(
        low=0,
        high=num_nodes,
        size=(max(2, math.ceil(num_nodes * 0.1)),),
        generator=generator,
        device=device,
    )
    augmented_data = AtomicDataDict.without_nodes(data, node_idx)
    augmented_data_list = [augmented_data[key] for key in fields]
    augmented_fx_model = _nequip_make_fx(model, augmented_data_list + extra_inputs)

    # 3. finally with the original input data again
    fx_model = _nequip_make_fx(model, test_data_list + extra_inputs)

    # then test last two (which corresponds to different shapes)
    check_make_fx_diff(augmented_fx_model, fx_model)

    # and return the last one
    return fx_model


def _nequip_make_fx(model, inputs):
    return make_fx(
        model,
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
        _error_on_data_dependent_ops=True,
    )(*inputs)


def highlight_code_differences(code1, code2):
    differ = difflib.Differ()
    diff = list(differ.compare(code1.splitlines(), code2.splitlines()))

    highlighted = []
    for line in diff:
        if line.startswith("  "):  # unchanged line
            highlighted.append(line[2:])
        elif line.startswith("- "):  # removed line
            highlighted.append(f"\033[91m{line[2:]}\033[0m")  # red
        elif line.startswith("+ "):  # Added line
            highlighted.append(f"\033[92m{line[2:]}\033[0m")  # green
        elif line.startswith("? "):  # line with changes
            continue  # skip the change markers

    return "\n".join(highlighted)


def check_make_fx_diff(fx_model_1, fx_model_2):
    if fx_model_1.code != fx_model_2.code:
        # the following is commented to prevent obscuring the error message below
        # devs can uncomment for diagonosing shape specializations
        # print(highlight_code_differences(fx_model_1.code, fx_model_2.code))
        dump_dir = str(os.getcwd()) + "/nequip_fx_dump"
        os.mkdir(dump_dir)
        with open(dump_dir + "/fx_model_1.txt", "w") as f:
            f.write(fx_model_1.code)
        with open(dump_dir + "/fx_model_2.txt", "w") as f:
            f.write(fx_model_2.code)
        raise RuntimeError(
            f"An unexpected internal error has occurred (the fx'ed models for different input shapes do not agree) -- please report this issue on the NequIP GitHub, and upload the files in {dump_dir}."
        )
