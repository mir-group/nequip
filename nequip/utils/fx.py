# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import math
import torch

from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import core_aten_decompositions

from nequip.data import AtomicDataDict
import contextlib
import difflib
import uuid
import os
from typing import List, Optional


@contextlib.contextmanager
def fx_duck_shape(enabled: bool):
    """
    For our use of `make_fx` to unfold the autograd graph, we must set the following `use_duck_shape` parameter to `False` (it's `True` by default).
    It forces dynamic batch dims (num_frames, num_atoms, num_edges) to shape specialize if the batch dim is the same as that of a static dim.
    E.g. in training, shape specialization would occur if a weight tensor has a dimension with shape (16,) and we use a batch size of 16 (so the dynamic batch dim `num_frames` is 16) because of the duck shaping.
    """
    # save previous state
    init_duck_shape = torch.fx.experimental._config.use_duck_shape
    # set mode variables
    torch.fx.experimental._config.use_duck_shape = enabled
    try:
        yield
    finally:
        # restore state
        torch.fx.experimental._config.use_duck_shape = init_duck_shape


def nequip_make_fx(
    model: torch.nn.Module,
    data: AtomicDataDict.Type,
    fields: List[str],
    extra_inputs: Optional[List[torch.Tensor]] = None,
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
    # === preprocess data ===
    data = data.copy()
    extra_inputs = [] if extra_inputs is None else extra_inputs
    data = {key: data[key] for key in fields}

    # we do it twice
    # 1. once with the original input data
    test_data_list = [data[key] for key in fields]
    fx_model = _nequip_make_fx(model, test_data_list + extra_inputs)

    # 2. once with augmented data where batch dims are different
    # there are two cases:
    # - if there are no batches, num_atoms, num_edges are different (num_frames assumed to be 1)
    # - else all three of num_frames, num_atoms, num_edges are differnt
    device = data[AtomicDataDict.POSITIONS_KEY].device
    # get global seed to construct generator for reproducibility
    generator = torch.Generator(device).manual_seed(seed)
    single_frame = AtomicDataDict.frame_from_batched(data, 0)
    num_nodes = AtomicDataDict.num_nodes(single_frame)
    node_idx = torch.randint(
        low=0,
        high=num_nodes,
        size=(max(2, math.ceil(num_nodes * 0.1)),),
        generator=generator,
        device=device,
    )
    augmented_data = AtomicDataDict.without_nodes(single_frame, node_idx)
    if AtomicDataDict.BATCH_KEY in data:
        augmented_data = AtomicDataDict.batched_from_list([data, augmented_data])
    augmented_data = [augmented_data[key] for key in fields]
    augmented_fx_model = _nequip_make_fx(model, augmented_data + extra_inputs)
    del augmented_data, node_idx, single_frame

    # because we use different batch dims for each fx model,
    # check that the fx graphs are identical to ensure that `make_fx` didn't shape-specialize
    check_make_fx_diff(fx_model, augmented_fx_model, fields)

    # clean up
    torch.cuda.empty_cache()

    return fx_model


def _nequip_make_fx(model, inputs):
    with fx_duck_shape(False):
        return make_fx(
            model,
            # see below for explanation on decomposition table
            decomposition_table=core_aten_decompositions(),
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            _error_on_data_dependent_ops=True,
        )(*[i.clone() for i in inputs])

    # from PT 2.9.1 to PT 2.10.0, we get errors during training such as
    # RuntimeError: derivative for aten::silu_backward is not implemented
    # this is because of the double backwards and whether the fx graph is decomposed.
    # relevant lines in PT 2.9.1: https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/fx/experimental/proxy_tensor.py#L898
    # PT 2.10.0: https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/fx/experimental/proxy_tensor.py#L1044
    # specifically autograd_would_have_decomposed(func, flat_args_kwargs)
    # explanation: https://github.com/pytorch/pytorch/blob/5a48148c1ab83c1e3779283d904ba5744bbe8eb3/torch/utils/_python_dispatch.py#L811

    # to overcome this problem, we just always decompose
    # minimal testing indicated that it shouldn't be a problem to always decompose


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


def check_make_fx_diff(fx_model_1, fx_model_2, fields: List[str]):
    if fx_model_1.code != fx_model_2.code:
        # the following is commented to prevent obscuring the error message below
        # devs can uncomment for diagonosing shape specializations
        # print(highlight_code_differences(fx_model_1.code, fx_model_2.code))
        dump_dir = str(os.getcwd()) + "/nequip_fx_dump_" + str(uuid.uuid4())
        os.mkdir(dump_dir)
        with open(dump_dir + "/fx_model_1.txt", "w") as f:
            f.write(f"# Argument order:\n{fields} + extra_inputs\n")
            f.write(fx_model_1.code)
        with open(dump_dir + "/fx_model_2.txt", "w") as f:
            f.write(f"# Argument order:\n{fields} + extra_inputs\n")
            f.write(fx_model_2.code)
        raise RuntimeError(
            f"An unexpected internal error has occurred (the fx'ed models for different input shapes do not agree) -- please report this issue on the NequIP GitHub, and upload the files in {dump_dir}."
        )
