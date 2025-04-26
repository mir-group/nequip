from nequip.model.utils import (
    _EAGER_MODEL_KEY,
    _TRAIN_TIME_COMPILE_KEY,
    _COMPILE_MODE_OPTIONS,
    get_current_compile_mode,
    override_model_compile_mode,
)


def test_single_override():
    baseline = get_current_compile_mode()

    new_mode = _TRAIN_TIME_COMPILE_KEY
    assert new_mode in _COMPILE_MODE_OPTIONS

    with override_model_compile_mode(new_mode):
        assert get_current_compile_mode() == new_mode

    assert get_current_compile_mode() == baseline


def test_nested_override_ignored():
    """If an override is already active, an inner override is ignored."""
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
