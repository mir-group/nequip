from nequip.train.early_stopping import EarlyStopping


def test_upper():

    es = EarlyStopping(upper_bounds={"u": 3})

    stop, stop_args, debug_args = es({"u": 2})
    assert not stop, "wrong upper bound"

    stop, stop_args, debug_args = es({"u": 4})
    assert stop, "wrong upper bound"


def test_lower():

    es = EarlyStopping(lower_bounds={"u": 3})

    stop, stop_args, debug_args = es({"u": 2})
    assert stop, "wrong lower bound"

    stop, stop_args, debug_args = es({"u": 4})
    assert not stop, "wrong lower bound"


def test_plateau():

    es = EarlyStopping(patiences={"u": 3}, delta={"u": 0.1})

    stop, stop_args, debug_args = es({"u": 2.9})
    stop, stop_args, debug_args = es({"u": 2.7})
    stop, stop_args, debug_args = es({"u": 2.5})
    stop, stop_args, debug_args = es({"u": 2.3})
    assert not stop, "wrong patience setup"

    stop, stop_args, debug_args = es({"u": 2.5})
    stop, stop_args, debug_args = es({"u": 2.6})
    stop, stop_args, debug_args = es({"u": 2.5})
    stop, stop_args, debug_args = es({"u": 2.5})
    stop, stop_args, debug_args = es({"u": 2.5})
    assert stop, "wrong patience setup"
