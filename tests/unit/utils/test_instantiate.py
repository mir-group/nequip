import pytest
import yaml

from nequip.utils import instantiate


simple_default = {"b": 1, "d": 31}


class SimpleExample:
    def __init__(self, a, b=simple_default["b"], d=simple_default["d"]):
        self.a = a
        self.b = b
        self.d = d


nested_default = {"d": 37}


class NestedExample:
    def __init__(self, cls_c, a, cls_c_kwargs={}, d=nested_default["d"]):
        self.c_obj = cls_c(**cls_c_kwargs)
        self.a = a
        self.d = d


def assert_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            assert_dict(v)
        elif isinstance(v, str):
            assert k == v


@pytest.mark.parametrize("positional_args", [dict(a=3, b=4), dict(a=5), dict()])
@pytest.mark.parametrize("optional_args", [dict(a=3, b=4), dict(a=5), dict()])
@pytest.mark.parametrize("all_args", [dict(a=6, b=7), dict(a=8), dict()])
@pytest.mark.parametrize("prefix", [True, False])
def test_simple_init(positional_args, optional_args, all_args, prefix):

    union = {}
    union.update(all_args)
    union.update(optional_args)
    union.update(positional_args)
    if "a" not in union:
        return

    # decorate test with prefix
    _all_args = (
        {"simple_example_" + k: v for k, v in all_args.items()} if prefix else all_args
    )

    # check key mapping is correct
    km, params = instantiate(
        builder=SimpleExample,
        prefix="simple_example",
        positional_args=positional_args,
        optional_args=optional_args,
        all_args=_all_args,
        return_args_only=True,
    )
    for t in km:
        for k, v in km[t].items():
            assert k in locals()[t + "_args"]
            if prefix and t == "all":
                assert v == "simple_example_" + k
            else:
                assert v == k
    km, _ = instantiate(
        builder=SimpleExample,
        prefix="simple_example",
        positional_args=positional_args,
        all_args=params,
        return_args_only=True,
    )
    assert_dict(km)

    # check whether it gets the priority right
    a1, params = instantiate(
        builder=SimpleExample,
        prefix="simple_example",
        positional_args=positional_args,
        optional_args=optional_args,
        all_args=_all_args,
    )
    assert a1.a == union["a"]
    if "b" in union:
        assert a1.b == union["b"]
    else:
        assert a1.b == simple_default["b"]
    for k in params:
        if k in simple_default:
            assert params[k] == union.get(k, simple_default[k])

    # check whether the return value is right
    a2 = SimpleExample(**positional_args, **params)
    assert a1.a == a2.a
    assert a1.b == a2.b


def test_prefix_priority():

    args = {"prefix_a": 3, "a": 4}

    a, params = instantiate(
        builder=SimpleExample,
        prefix="prefix",
        all_args=args,
    )
    assert a.a == 3


@pytest.mark.parametrize("optional_args", [dict(a=3, b=4), dict(a=5), dict()])
@pytest.mark.parametrize("all_args", [dict(a=6, b=7), dict(a=8), dict()])
@pytest.mark.parametrize("prefix", [True, False])
def test_nested_kwargs(optional_args, all_args, prefix):

    union = {}
    union.update(all_args)
    union.update(optional_args)
    if "a" not in union:
        return
    c, params = instantiate(
        builder=NestedExample,
        prefix="prefix",
        positional_args={"cls_c": SimpleExample},
        optional_args=optional_args,
        all_args=all_args,
    )


def test_default():
    """
    check the default value will not contaminate the other class
    """

    c, params = instantiate(
        builder=NestedExample,
        prefix="prefix",
        positional_args={"cls_c": SimpleExample},
        optional_args={"a": 11},
    )
    c.d = nested_default["d"]
    c.c_obj.d = simple_default["d"]


class A:
    def __init__(self, cls_a, cls_a_kwargs):
        self.a_obj = cls_a(**cls_a_kwargs)


class B:
    def __init__(self, cls_b, cls_b_kwargs):
        self.b_obj = cls_b(**cls_b_kwargs)


class C:
    def __init__(self, cls_c, cls_c_kwargs):  # noqa
        self.c_obj = c_cls(**c_cls_kwargs)  # noqa


def test_deep_nests():
    all_args = {"a": 101, "b": 103, "c": 107}
    obj, params = instantiate(
        builder=NestedExample,
        optional_args={"cls_c": A, "cls_a": B, "cls_b": SimpleExample},
        all_args=all_args,
    )

    print(yaml.dump(params))
    assert obj.c_obj.a_obj.b_obj.a == all_args["a"]
    assert obj.c_obj.a_obj.b_obj.b == all_args["b"]
    assert obj.c_obj.a_obj.b_obj.d == simple_default["d"]
    assert obj.d == nested_default["d"]

    obj = NestedExample(**params)
    assert obj.c_obj.a_obj.b_obj.a == all_args["a"]
    assert obj.c_obj.a_obj.b_obj.b == all_args["b"]
    assert obj.c_obj.a_obj.b_obj.d == simple_default["d"]
    assert obj.d == nested_default["d"]

    km, params = instantiate(
        builder=NestedExample,
        optional_args={"cls_c": A, "cls_a": B, "cls_b": SimpleExample},
        all_args=all_args,
        return_args_only=True,
    )
    print(yaml.dump(km))

    # check the key mapping is unique for
    km, _ = instantiate(
        builder=NestedExample, optional_args=params, return_args_only=True
    )
    assert_dict(km)


def test_recursion_nests():
    with pytest.raises(RuntimeError) as excinfo:
        b, params = instantiate(
            builder=A,
            positional_args={"cls_a": B},
            optional_args={"cls_b": A},
        )
    assert "cyclic" in str(excinfo.value)
    print(excinfo)


def test_cyclic_nests():
    with pytest.raises(RuntimeError) as excinfo:
        c, params = instantiate(
            builder=A,
            positional_args={"cls_a": B},
            optional_args={"cls_b": C},
            all_args={"cls_c": A},
        )
    assert "cyclic" in str(excinfo.value)
    print(excinfo, "hello")


class BadKwargs1:
    def __init__(self, thing_kwargs={}):
        pass


class BadKwargs2:
    def __init__(self, thing="a string", thing_kwargs={}):
        pass


def test_bad_kwargs():
    with pytest.raises(KeyError):
        _ = instantiate(BadKwargs1)
    with pytest.raises(ValueError):
        _ = instantiate(BadKwargs2)
