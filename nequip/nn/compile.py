# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from nequip.data import AtomicDataDict
from nequip.data._key_registry import get_dynamic_shapes
from .graph_model import GraphModel
from ._graph_mixin import GraphModuleMixin
from nequip.utils.dtype import (
    test_model_output_similarity_by_dtype,
    _pt2_compile_error_message,
)
from nequip.utils.fx import nequip_make_fx
from nequip.utils.dtype import dtype_to_name
from typing import Dict, Sequence, List, Optional, Any


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
    """Like ``ListInputOutputWrapper``, but also updates the model with state dict entries before each ``forward``."""

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
        # have to do it this way and not using state_dict directly for autograd reasons
        # the `.data` part is important
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(state_dict[name])
            for name, buffer in self.model.named_buffers():
                buffer.data.copy_(state_dict[name])
        output_dict = self.model(input_dict)
        return _list_from_dict(self.output_keys, output_dict)


class CompileGraphModel(GraphModel):
    """Wrapper that uses ``torch.compile`` to optimize the wrapped module while allowing it to be trained."""

    def __init__(
        self,
        model: GraphModuleMixin,
        model_config: Optional[Dict[str, str]] = None,
        model_input_fields: Dict[str, Any] = {},
    ) -> None:
        super().__init__(model, model_config, model_input_fields)
        # save model param and buffer names
        self.weight_names = [n for n, _ in self.model.named_parameters()]
        self.buffer_names = [n for n, _ in self.model.named_buffers()]
        # these will be updated when the model is compiled
        self._compiled_model = ()
        self.input_fields = None
        self.output_fields = None

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

        # === compile ===
        # compilation happens on the first data pass when there are at least two atoms (hard to pre-emp pathological data)
        if not self._compiled_model:
            # == get input and output fields ==
            # use intersection of data keys and GraphModel input/outputs, which assumes
            # - correctness of irreps registration system
            # - all input `data` batches have the same keys, and contain all necessary inputs and reference labels (outputs)
            self.input_fields = sorted(list(data.keys() & self.model_input_fields))
            self.output_fields = sorted(
                list(data.keys() & self.model.irreps_out.keys())
            )

            # == preprocess model and make_fx ==
            model_to_trace = ListInputOutputStateDictWrapper(
                model=self.model,
                input_keys=self.input_fields,
                output_keys=self.output_fields,
                state_dict_keys=self.weight_names + self.buffer_names,
            )

            weights, buffers = self._get_weights_buffers()
            fx_model = nequip_make_fx(
                model=model_to_trace,
                data=data,
                fields=self.input_fields,
                extra_inputs=weights + buffers,
            )

            # == export with dynamic shape specification ==
            # TODO: (maybe) include range for dynamic dims
            batch_map = {
                "graph": torch.export.dynamic_shapes.Dim("graph"),
                "node": torch.export.dynamic_shapes.Dim("node"),
                "edge": torch.export.dynamic_shapes.Dim("edge"),
            }
            dynamic_shapes = get_dynamic_shapes(
                self.input_fields + self.weight_names + self.buffer_names, batch_map
            )
            exported = torch.export.export(
                fx_model,
                (*([data[k] for k in self.input_fields] + weights + buffers),),
                dynamic_shapes=dynamic_shapes,
            )
            del weights, buffers

            # == compile exported program ==
            # see https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html#running-the-exported-program
            # TODO: compile options
            self._compiled_model = (
                torch.compile(
                    exported.module(),
                    dynamic=True,
                    fullgraph=True,
                ),
            )
            # NOTE: the compiled model is wrapped in a tuple so that it's not registered and saved in the state dict -- this is necessary to enable `GraphModel` to load `CompileGraphModel` state dicts
            # see https://discuss.pytorch.org/t/saving-nn-module-to-parent-nn-module-without-registering-paremeters/132082/6

            # run original model and compiled model with data to sanity check
            test_model_output_similarity_by_dtype(
                self._compiled_forward,
                self.model,
                {k: data[k] for k in self.input_fields},
                dtype_to_name(self.model_dtype),
                fields=self.output_fields,
                error_message=_pt2_compile_error_message,
            )

        # === run compiled model ===
        out_dict = self._compiled_forward(data)
        to_return = data.copy()
        to_return.update(out_dict)
        return to_return

    def _compiled_forward(self, data):
        # run compiled model with data
        weights, buffers = self._get_weights_buffers()
        data_list = _list_from_dict(self.input_fields, data)
        out_list = self._compiled_model[0](*(data_list + weights + buffers))
        out_dict = _list_to_dict(self.output_fields, out_list)
        return out_dict

    def _get_weights_buffers(self):
        # get weights and buffers from trainable model
        weight_dict = dict(self.model.named_parameters())
        weights = [weight_dict[name] for name in self.weight_names]
        buffer_dict = dict(self.model.named_buffers())
        buffers = [buffer_dict[name] for name in self.buffer_names]
        return weights, buffers
