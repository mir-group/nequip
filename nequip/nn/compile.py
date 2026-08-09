# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import os
import torch

from nequip.data import AtomicDataDict
from .graph_model import GraphModel
from ._graph_mixin import GraphModuleMixin
from nequip.utils.dtype import (
    test_model_output_similarity_by_dtype,
    _pt2_compile_error_message,
)
from nequip.utils.fx import nequip_make_fx
from nequip.utils.dtype import dtype_to_name
from typing import Dict, Sequence, List, Optional, Any, Final
from torch.func import functional_call

_NEQUIP_NODE_PARALLEL_COMPILE: Final[bool] = os.environ.get(
    "NEQUIP_NODE_PARALLEL_COMPILE", ""
).lower() in (
    "1",
    "true",
    "yes",
    "y",
)


def _local_rank_layout() -> tuple[int, int, int]:
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    local_size = int(
        os.environ.get(
            "SLURM_NTASKS_PER_NODE",
            torch.cuda.device_count() if torch.cuda.is_available() else 1,
        )
    )
    # derived from the global rank rather than SLURM_LOCALID so that it agrees by
    # construction with the contiguous `torch.distributed.new_subgroups` split;
    # exactly one local rank 0 per group, whatever the task layout
    return rank, rank % local_size, local_size


def _list_to_dict(
    keys: Sequence[str], args: List[torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return {key: arg for key, arg in zip(keys, args)}


def _list_from_dict(
    keys: Sequence[str], data: Dict[str, torch.Tensor]
) -> List[torch.Tensor]:
    return [data[key] for key in keys]


class ListInputOutputWrapper(torch.nn.Module):
    """
    Wraps a ``torch.nn.Module`` that takes and returns ``Dict[str, torch.Tensor]`` to have it take and return ``Sequence[torch.Tensor]`` for specified input and output fields.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_keys: Sequence[str],
        output_keys: Sequence[str],
    ):
        super().__init__()
        self.model = model
        self.input_keys = list(input_keys)
        self.output_keys = list(output_keys)

    def forward(self, *args: torch.Tensor) -> List[torch.Tensor]:
        inputs = _list_to_dict(self.input_keys, args)
        outputs = self.model(inputs)
        return _list_from_dict(self.output_keys, outputs)


class DictInputOutputWrapper(torch.nn.Module):
    """
    Wraps a model that takes and returns ``Sequence[torch.Tensor]`` to have it take and return ``Dict[str, torch.Tensor]`` for specified input and output fields (i.e. the opposite of ``ListInputOutputWrapper``).
    """

    def __init__(self, model, input_keys: List[str], output_keys: List[str]):
        super().__init__()
        self.model = model
        self.input_keys = input_keys
        self.output_keys = output_keys

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        inputs = _list_from_dict(self.input_keys, data)
        with torch.inference_mode():
            outputs = self.model(inputs)
        return _list_to_dict(self.output_keys, outputs)


class ListInputOutputStateDictWrapper(ListInputOutputWrapper):
    """Like ``ListInputOutputWrapper``, but also updates the model with state dict entries before each ``forward`` using ``functional_call``."""

    def __init__(
        self,
        model: torch.nn.Module,
        input_keys: Sequence[str],
        output_keys: Sequence[str],
        state_dict_keys: Sequence[str],
    ):
        super().__init__(model, input_keys, output_keys)
        self.state_dict_keys = state_dict_keys

    def forward(self, *args: torch.Tensor) -> List[torch.Tensor]:
        # won't check that `args` is of the correct length
        input_dict = _list_to_dict(self.input_keys, args[: len(self.input_keys)])
        state_dict = _list_to_dict(self.state_dict_keys, args[len(self.input_keys) :])
        # use functional_call to avoid in-place modification
        output_dict = functional_call(self.model, state_dict, args=(input_dict,))
        return _list_from_dict(self.output_keys, output_dict)


class CompileGraphModel(GraphModel):
    """Wrapper that uses ``torch.compile`` to optimize the wrapped module while allowing it to be trained.

    The cache is keyed by input signature (input keys only).
    For each input signature, the eager model is run to determine the output keys, and then a compiled model is created for that input/output combination.
    The compiled model and output keys are stored together in the cache.
    """

    is_compile_graph_model: Final[bool] = True
    # ^ to identify `GraphModel` types from `nequip-package`d models (see https://pytorch.org/docs/stable/package.html#torch-package-sharp-edges)

    def __init__(
        self,
        model: GraphModuleMixin,
        model_config: Optional[Dict[str, str]] = None,
        model_input_fields: Dict[str, Any] = {},
    ) -> None:
        super().__init__(model, model_config, model_input_fields)
        # cache for multiple compiled variants based on input key signatures
        # cache structure: {input_signature: (compiled_model, output_fields)}
        # NOTE: the cache dict is wrapped in a tuple so that it's not registered and saved in the state dict -- this is necessary to enable `GraphModel` to load `CompileGraphModel` state dicts
        # see https://discuss.pytorch.org/t/saving-nn-module-to-parent-nn-module-without-registering-paremeters/132082/6
        self._compiled_cache = ({},)
        # weights and buffers should be done lazily because model modification can happen after instantiation
        # such that parameters and buffers may change between class instantiation and the lazy compilation in the `forward`
        self.weight_names = None
        self.buffer_names = None

    def _get_input_signature(self, data: AtomicDataDict.Type) -> tuple:
        """Compute a hashable signature for the input keys.

        The unique set of input keys determines a unique set of output keys when run through the model,
        so we only need the input keys for the cache lookup signature.

        Uses intersection of data keys and GraphModel inputs, which assumes:
        - correctness of irreps registration system
        - this particular batch contains all necessary inputs for this variant
        """
        input_keys = tuple(sorted(data.keys() & self.model_input_fields))
        return input_keys

    def _distributed_warm_compile(self, fn, label: str = "first compile") -> None:
        """One cold compile per node, then one parallel warm turn within the node.

        Opt-in via ``NEQUIP_NODE_PARALLEL_COMPILE=1`` (or ``true``/``yes``/``y``);
        off by default, where every rank compiles concurrently as usual.

        Each rank still needs its own in-process ``torch.compile`` wrapper: filesystem
        cache skips Inductor/Triton *codegen* on a cache hit; while ``make_fx`` + Dynamo
        still run on every rank. ``LOCAL_RANK`` ``0`` on each node cold-compiles (one per
        node, parallel across nodes) into the shared cache, then local ranks ``1..N-1``
        compile in one parallel turn (as guaranteed cache hits).
        """
        if not (
            _NEQUIP_NODE_PARALLEL_COMPILE
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            fn()
            return

        import sys
        import time

        rank, local_rank, local_size = _local_rank_layout()
        node_group = self._node_process_group(local_size)

        t0 = time.time()
        if local_rank == 0:
            # one marker per node: the expensive/hang-prone cold compile is starting here
            print(
                f"[compile] {label}: cold compile (node warm)",
                file=sys.stderr,
                flush=True,
            )
            fn()
            print(
                f"[compile-timing] rank {rank} (local0) cold compile: {time.time() - t0:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        # local ranks 1..N-1 must wait for local0's codegen to land in the shared cache
        torch.distributed.barrier(group=node_group)
        t_warm = time.time()
        # warm ranks are cache hits (local0's codegen just landed): one parallel turn
        if local_rank >= 1:
            fn()
        torch.distributed.barrier(group=node_group)
        if local_rank == 0:
            print(
                f"[compile-timing] rank {rank} (local0) warm phase "
                f"({local_size - 1} warm ranks): {time.time() - t_warm:.1f}s; "
                f"total {time.time() - t0:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    def _node_process_group(self, local_size: int):
        """Cached intranode process group for per-node compile warm-up."""
        cache_key = f"_nequip_compile_pg_{local_size}"
        if getattr(self, cache_key, None) is None:
            # ``new_subgroups`` raises on a world size that is not a whole number of nodes
            setattr(self, cache_key, torch.distributed.new_subgroups(local_size)[0])
        return getattr(self, cache_key)

    def _compile_variant(
        self,
        data: AtomicDataDict.Type,
        input_signature: tuple,
        cache: dict,
    ) -> None:
        """Trace, compile, sanity-check, and store one cache entry for ``input_signature``."""
        if self.weight_names is None:
            self.weight_names = [n for n, _ in self.model.named_parameters()]
            self.buffer_names = [n for n, _ in self.model.named_buffers()]

        # == get input fields for this variant ==
        input_fields = list(input_signature)

        # == run eager model to determine output fields ==
        eager_output = super().forward(data.copy())
        output_fields = tuple(sorted(eager_output.keys()))
        del eager_output

        # == preprocess model and make_fx ==
        model_to_trace = ListInputOutputStateDictWrapper(
            model=self.model,
            input_keys=input_fields,
            output_keys=output_fields,
            state_dict_keys=self.weight_names + self.buffer_names,
        )

        weights, buffers = self._get_weights_buffers()
        fx_model = nequip_make_fx(
            model=model_to_trace,
            data=data,
            fields=input_fields,
            extra_inputs=weights + buffers,
        )
        del weights, buffers

        # == compile exported program ==
        # see https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html#running-the-exported-program
        # TODO: compile options
        compiled_model = torch.compile(
            fx_model,
            dynamic=True,
            fullgraph=False,
        )

        # store in cache: (compiled_model, output_fields)
        cache[input_signature] = (compiled_model, output_fields)

        # run original model and compiled model with data to sanity check
        def compiled_forward_for_test(data_test):
            return self._compiled_forward(
                data_test, compiled_model, input_fields, output_fields
            )

        # only test output fields that are present in data (i.e. labels are present)
        test_fields = sorted(set(output_fields) & data.keys())
        test_model_output_similarity_by_dtype(
            compiled_forward_for_test,
            self.model,
            {k: data[k] for k in input_fields},
            dtype_to_name(self.model_dtype),
            fields=test_fields,
            error_message=_pt2_compile_error_message,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # short-circuit if one of the batch dims is 1 (0 would be an error)
        # this is related to the 0/1 specialization problem
        # see https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ&tab=t.0#heading=h.ez923tomjvyk
        # we just need something that doesn't have a batch dim of 1 to `make_fx` or else it'll shape specialize
        # the models compiled for more batch_size > 1 data cannot be used for batch_size=1 data
        # (under specific cases related to the `PerTypeScaleShift` module)
        # for now we just make sure to always use the eager model when the data has any batch dims of 1
        if (
            AtomicDataDict.num_nodes(data) < 2
            or AtomicDataDict.num_frames(data) < 2
            or AtomicDataDict.num_edges(data) < 2
        ):
            # use parent class's forward
            return super().forward(data)

        # === get or compile variant for this input signature ===
        # compilation happens lazily when we encounter a new combination of input keys
        input_signature = self._get_input_signature(data)
        cache = self._compiled_cache[0]

        if input_signature not in cache:
            # get weight names and buffers (only once on first compilation)
            if not getattr(self, "_nequip_serialized_first_compile", False):

                def _first_compile():
                    self._compile_variant(data, input_signature, cache)

                self._distributed_warm_compile(_first_compile)
                self._nequip_serialized_first_compile = True
            else:
                self._compile_variant(data, input_signature, cache)

        # === run compiled model for this variant ===
        compiled_model, output_fields = cache[input_signature]
        out_dict = self._compiled_forward(
            data, compiled_model, input_signature, output_fields
        )
        to_return = data.copy()
        to_return.update(out_dict)
        return to_return

    def _compiled_forward(self, data, compiled_model, input_fields, output_fields):
        # run compiled model with data
        weights, buffers = self._get_weights_buffers()
        data_list = _list_from_dict(input_fields, data)
        out_list = compiled_model(*(data_list + weights + buffers))
        out_dict = _list_to_dict(output_fields, out_list)
        return out_dict

    def _get_weights_buffers(self):
        # get weights and buffers from trainable model
        weight_dict = dict(self.model.named_parameters())
        weights = [weight_dict[name] for name in self.weight_names]
        buffer_dict = dict(self.model.named_buffers())
        buffers = [buffer_dict[name] for name in self.buffer_names]
        return weights, buffers
