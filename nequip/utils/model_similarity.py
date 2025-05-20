import torch
from nequip.utils import floating_point_tolerance
from nequip.data import AtomicDataDict

from typing import Callable, Optional, Union, List

def _pt2_compile_error_message(key, tol, err, absval, model_dtype):
    return f"Compilation check MaxAbsError: {err:.6g} (tol: {tol}) for field `{key}`. This assert was triggered because the outputs of an eager model and a compiled model numerically differ above the specified tolerance. This may indicate a compilation error (bug) specific to certain machines and installation environments, or may be an artefact of poor initialization if the error is close to the tolerance. Note that the largest absolute (MaxAbs) entry of the model prediction is {absval:.6g} -- you can use this detail to discern if it is a numerical artefact (the errors could be large because the MaxAbs value is very large) or a more fundamental compilation error. Raise a GitHub issue if you believe it is a compilation error or are unsure. If you are confident that it is purely numerical, and want to bypass the tolerance check, you may set the following environment variables: `NEQUIP_FLOAT64_MODEL_TOL`, `NEQUIP_FLOAT32_MODEL_TOL`, or `NEQUIP_TF32_MODEL_TOL`, depending on the model dtype you are using (which is currently {model_dtype}) and whether TF32 is on."


def _default_error_message(key, tol, err, absval, model_dtype):
    return f"MaxAbsError: {err:.6g} (tol: {tol} for {model_dtype} model) for field `{key}`. MaxAbs value: {absval:.6g}."

# for `test_model_output_similarity`, we perform evaluation `_NUM_EVAL_TRIALS` times to account for numerical randomness in the model
_NUM_EVAL_TRIALS = 5

def test_model_output_similarity_by_dtype(
    model1: Callable,
    model2: Callable,
    data: AtomicDataDict.Type,
    model_dtype: Union[str, torch.dtype],
    fields: Optional[List[str]] = None,
    error_message: Callable = _default_error_message,
):
    """
    Assumptions and behavior:
    - `model1` and `model2` have signature `AtomicDataDict -> AtomicDataDict`
    - if `fields` are not provided, the function will loop over `model1`'s output keys
    """
    tol = floating_point_tolerance(model_dtype)

    # do one evaluation to figure out the fields if not provided
    if fields is None:
        fields = model1(data.copy()).keys()

    # perform `_NUM_EVAL_TRIALS` evaluations with each model and average to account for numerical randomness
    out1_list, out2_list = {k: [] for k in fields}, {k: [] for k in fields}
    for _ in range(_NUM_EVAL_TRIALS):
        out1 = model1(data.copy())
        out2 = model2(data.copy())
        for k in fields:
            out1_list[k].append(out1[k].detach().double())
            out2_list[k].append(out2[k].detach().double())
        del out1, out2

    for k in fields:
        t1, t2 = (
            torch.mean(torch.stack(out1_list[k], -1), -1),
            torch.mean(torch.stack(out2_list[k], -1), -1),
        )
        err = torch.max(torch.abs(t1 - t2)).item()
        absval = t1.abs().max().item()

        assert torch.allclose(t1, t2, atol=tol, rtol=tol), error_message(
            k, tol, err, absval, model_dtype
        )

        del t1, t2, err, absval