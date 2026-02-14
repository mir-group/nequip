import contextlib

import pytest
import torch

from nequip.model.utils import (
    _EAGER_MODEL_KEY,
    _TRAIN_TIME_COMPILE_KEY,
    _COMPILE_MODE_OPTIONS,
    fresh_model_builder_context,
    get_current_compile_mode,
    model_builder,
    override_model_compile_mode,
)


class _ToyModel(torch.nn.Module):
    """Minimal module with irreps attributes for model_builder tests."""

    def __init__(self):
        super().__init__()
        self.irreps_in = {}
        self.irreps_out = {}

    def forward(self, data):
        return data


class _ToyWrapper(torch.nn.Module):
    """Minimal wrapper matching the constructor shape used by model_builder.

    This is a lightweight stand-in for GraphModel/CompileGraphModel in unit tests.
    """

    def __init__(self, model, model_config=None, model_input_fields=None):
        super().__init__()
        self.model = model
        self.model_config = model_config
        self.model_input_fields = model_input_fields
        # pass through irreps so nested wrapped outputs can be wrapped again
        self.irreps_in = getattr(model, "irreps_in", {})
        self.irreps_out = getattr(model, "irreps_out", {})


def test_single_override():
    """``override_model_compile_mode`` sets compile mode temporarily and restores it."""
    baseline = get_current_compile_mode()

    new_mode = _TRAIN_TIME_COMPILE_KEY
    assert new_mode in _COMPILE_MODE_OPTIONS

    with override_model_compile_mode(new_mode):
        assert get_current_compile_mode() == new_mode
        mode, is_override = get_current_compile_mode(return_override=True)
        assert mode == new_mode
        assert is_override

    assert get_current_compile_mode() == baseline


def test_nested_override_ignored():
    """If an ``override_model_compile_mode`` is already active, an inner one is ignored."""
    outer_mode = _TRAIN_TIME_COMPILE_KEY
    inner_mode = _EAGER_MODEL_KEY
    assert outer_mode != inner_mode

    with override_model_compile_mode(outer_mode):
        # outer override in effect
        assert get_current_compile_mode() == outer_mode

        # inner attempt should not supersede the outer one
        with override_model_compile_mode(inner_mode):
            assert get_current_compile_mode() == outer_mode

        # still outer after inner context exits
        assert get_current_compile_mode() == outer_mode


def test_model_builder_requires_args_and_cleans_state(monkeypatch):
    """``model_builder`` enforces required args and resets internal state after each call."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    @model_builder(wrapper_class=_ToyWrapper, compile_wrapper_class=_ToyWrapper)
    def build_toy(**kwargs):
        return _ToyModel()

    good_kwargs = {
        "seed": 123,
        "model_dtype": "float32",
        "type_names": ["H"],
        "r_max": 1.0,
    }
    required_keys = ("seed", "model_dtype", "type_names")

    for key in required_keys:
        bad_kwargs = good_kwargs.copy()
        bad_kwargs.pop(key)
        with pytest.raises(AssertionError, match="mandatory model arguments"):
            build_toy(**bad_kwargs)

    model = build_toy(**good_kwargs)
    assert isinstance(model, _ToyWrapper)

    for key in required_keys:
        bad_kwargs = good_kwargs.copy()
        bad_kwargs.pop(key)
        with pytest.raises(AssertionError, match="mandatory model arguments"):
            build_toy(**bad_kwargs)


@pytest.mark.parametrize("use_fresh_context", [False, True])
def test_nested_builder_behavior_with_and_without_fresh_context(
    monkeypatch, use_fresh_context
):
    """``fresh_model_builder_context`` flips nested builder behavior from raw-inner to wrapped."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    @model_builder(wrapper_class=_ToyWrapper, compile_wrapper_class=_ToyWrapper)
    def inner_builder(**kwargs):
        return _ToyModel()

    @model_builder(wrapper_class=_ToyWrapper, compile_wrapper_class=_ToyWrapper)
    def outer_builder(use_fresh_context: bool, **kwargs):
        cm = (
            fresh_model_builder_context()
            if use_fresh_context
            else contextlib.nullcontext()
        )
        with cm:
            return inner_builder()

    outer = outer_builder(
        use_fresh_context=use_fresh_context,
        seed=123,
        model_dtype="float32",
        type_names=["H"],
        r_max=1.0,
    )
    assert isinstance(outer, _ToyWrapper)

    if use_fresh_context:
        assert isinstance(outer.model, _ToyWrapper)
        assert isinstance(outer.model.model, _ToyModel)
        assert outer.model.model_config["seed"] == 123
        assert outer.model.model_config["model_dtype"] == "float32"
        assert outer.model.model_config["type_names"] == ["H"]
    else:
        assert isinstance(outer.model, _ToyModel)


def test_model_builder_consumes_internal_kwargs(monkeypatch):
    """``model_builder`` consumes internal kwargs and forwards only user kwargs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    observed = {}

    @model_builder(wrapper_class=_ToyWrapper, compile_wrapper_class=_ToyWrapper)
    def build_toy(user_flag, **kwargs):
        observed["user_flag"] = user_flag
        observed["kwargs"] = dict(kwargs)
        return _ToyModel()

    model = build_toy(
        user_flag="keep-me",
        passthrough=3.14,
        seed=123,
        model_dtype="float32",
        compile_mode=_EAGER_MODEL_KEY,
        type_names=["H"],
        r_max=1.0,
    )

    assert isinstance(model, _ToyWrapper)
    assert observed["user_flag"] == "keep-me"
    assert observed["kwargs"]["passthrough"] == 3.14
    assert observed["kwargs"]["type_names"] == ["H"]
    assert "seed" not in observed["kwargs"]
    assert "model_dtype" not in observed["kwargs"]
    assert "compile_mode" not in observed["kwargs"]
