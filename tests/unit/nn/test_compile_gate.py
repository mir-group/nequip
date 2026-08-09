# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import os

import nequip.nn.compile as compile_module
from nequip.nn.compile import CompileGraphModel


def test_distributed_warm_compile_gate(monkeypatch):
    """The node-parallel path must be opt-in, and inert without a process group."""
    calls = []

    def fn():
        calls.append(1)

    # unset means off, so that the default is upstream's concurrent compile
    if "NEQUIP_NODE_PARALLEL_COMPILE" not in os.environ:
        assert compile_module._NEQUIP_NODE_PARALLEL_COMPILE is False

    # off -> straight through
    monkeypatch.setattr(compile_module, "_NEQUIP_NODE_PARALLEL_COMPILE", False)
    CompileGraphModel._distributed_warm_compile(object(), fn)

    # opted in, but no process group -> still straight through
    monkeypatch.setattr(compile_module, "_NEQUIP_NODE_PARALLEL_COMPILE", True)
    CompileGraphModel._distributed_warm_compile(object(), fn)

    assert len(calls) == 2
