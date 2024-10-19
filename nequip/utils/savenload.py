"""
utilities that involve file searching and operations (i.e. save/load)
"""

from typing import Union, Optional, Callable
import yaml


def load_callable(obj: Union[str, Callable], prefix: Optional[str] = None) -> Callable:
    """Load a callable from a name, or pass through a callable."""
    if callable(obj):
        pass
    elif isinstance(obj, str):
        if "." not in obj:
            # It's an unqualified name
            if prefix is not None:
                obj = prefix + "." + obj
            else:
                # You can't have an unqualified name without a prefix
                raise ValueError(f"Cannot load unqualified name {obj}.")
        obj = yaml.load(f"!!python/name:{obj}", Loader=yaml.Loader)
    else:
        raise TypeError
    assert callable(obj), f"{obj} isn't callable"
    return obj
