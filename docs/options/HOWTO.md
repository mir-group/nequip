Add this code to `auto_init.py`:

```python
f = open("auto_all_options.rst", "w")


def print_option(builder, file):
    print(f"!! {builder.__name__}", file=f)
    if inspect.isclass(builder):
        builder = builder.__init__
    sig = inspect.signature(builder)
    for k, v in sig.parameters.items():
        if k == "self":
            continue
        print(k, file=f)
        print(len(k) * "^", file=f)
        if v.default == inspect.Parameter.empty:
            print(f"    | Type:", file=f)
            print(
                f"    | Default: n/a\n",
                file=f,
            )
        else:
            typestr = type(v.default).__name__
            print(f"    | Type: {typestr}", file=f)
            print(
                f"    | Default: ``{str(v.default)}``\n",
                file=f,
            )
```

and call the function in every `instantiate`.