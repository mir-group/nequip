# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
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
    """Wrapper that uses ``torch.compile`` to optimize the wrapped module while allowing it to be trained."""

    is_compile_graph_model: Final[bool] = True
    # ^ to identify `GraphModel` types from `nequip-package`d models (see https://pytorch.org/docs/stable/package.html#torch-package-sharp-edges)

    def __init__(
        self,
        model: GraphModuleMixin,
        model_config: Optional[Dict[str, str]] = None,
        model_input_fields: Dict[str, Any] = {},
    ) -> None:
        super().__init__(model, model_config, model_input_fields)
        # cache for multiple compiled variants based on input/output key signatures
        # NOTE: the cache dict is wrapped in a tuple so that it's not registered and saved in the state dict -- this is necessary to enable `GraphModel` to load `CompileGraphModel` state dicts
        # see https://discuss.pytorch.org/t/saving-nn-module-to-parent-nn-module-without-registering-paremeters/132082/6
        self._compiled_cache = ({},)
        # weights and buffers should be done lazily because model modification can happen after instantiation
        # such that parameters and buffers may change between class instantiation and the lazy compilation in the `forward`
        self.weight_names = None
        self.buffer_names = None

    def _get_signature(self, data: AtomicDataDict.Type) -> tuple:
        """Compute a hashable signature for the input/output key combination.

        Uses intersection of data keys and GraphModel input/outputs, which assumes:
        - correctness of irreps registration system
        - this particular batch contains all necessary inputs and reference labels (outputs) for this variant
        """
        input_keys = tuple(sorted(data.keys() & self.model_input_fields))
        # `output_keys` relies on the fact that `data` contains the necessary output keys
        # e.g. `total_energy`, `forces`, `stress`
        output_keys = tuple(sorted(data.keys() & self.model.irreps_out.keys()))
        return (input_keys, output_keys)

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

        # === get or compile variant for this key signature ===
        # compilation happens lazily when we encounter a new combination of input/output keys
        signature = self._get_signature(data)
        cache = self._compiled_cache[0]

        if signature not in cache:
            # get weight names and buffers (only once on first compilation)
            if self.weight_names is None:
                self.weight_names = [n for n, _ in self.model.named_parameters()]
                self.buffer_names = [n for n, _ in self.model.named_buffers()]

            # == get input and output fields for this variant ==
            # extract from signature (which already computed the intersection of data keys and model input/output fields)
            input_fields = list(signature[0])
            output_fields = list(signature[1])

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
                fullgraph=True,
            )

            # store in cache
            cache[signature] = compiled_model

            # run original model and compiled model with data to sanity check
            def compiled_forward_for_test(data_test):
                return self._compiled_forward(data_test, compiled_model, signature)

            test_model_output_similarity_by_dtype(
                compiled_forward_for_test,
                self.model,
                {k: data[k] for k in input_fields},
                dtype_to_name(self.model_dtype),
                fields=output_fields,
                error_message=_pt2_compile_error_message,
            )

        # === run compiled model for this variant ===
        compiled_model = cache[signature]
        out_dict = self._compiled_forward(data, compiled_model, signature)
        to_return = data.copy()
        to_return.update(out_dict)
        return to_return

    def _compiled_forward(self, data, compiled_model, signature):
        # run compiled model with data
        input_fields = list(signature[0])
        output_fields = list(signature[1])
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
