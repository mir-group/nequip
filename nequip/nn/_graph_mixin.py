import random
from copy import deepcopy
from typing import Dict, Tuple, Callable, Any, Sequence, Optional, Union, Mapping
from collections import OrderedDict

import torch

from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.utils import instantiate


class GraphModuleMixin:
    def _init_irreps(
        self,
        irreps_in: Dict[str, Any] = {},
        my_irreps_in: Dict[str, Any] = {},
        required_irreps_in: Sequence[str] = [],
        irreps_out: Dict[str, Any] = {},
    ):
        """Set the data fields for this graph module.

        Args:
            irreps_in (dict): maps names of all input fields from previous modules or data to their corresponding irreps
            my_irreps_in (dict): maps names of fields to the irreps they must have for this graph module
            required_irreps_in: sequence of names of fields that must be present in ``irreps_in``, but that can have any irreps.
            irreps_out (dict): mapping names of fields that are modified/output by this graph module to their irreps.
        """
        # TODO: forward hook for checking input shapes?
        # Coerce
        irreps_in = {} if irreps_in is None else irreps_in
        irreps_in = AtomicDataDict._fix_irreps_dict(irreps_in)
        # question, what is this?
        my_irreps_in = AtomicDataDict._fix_irreps_dict(my_irreps_in)
        irreps_out = AtomicDataDict._fix_irreps_dict(irreps_out)
        # Confirm compatibility:
        # with my_irreps_in
        for k in my_irreps_in:
            if k in irreps_in and irreps_in[k] != my_irreps_in[k]:
                raise ValueError(
                    f"The given input irreps {irreps_in[k]} for field '{k}' is incompatible with this configuration {type(self)}; should have been {my_irreps_in[k]}"
                )
        # with required_irreps_in
        for k in required_irreps_in:
            if k not in irreps_in:
                raise ValueError(
                    f"This {type(self)} requires field '{k}' to be in irreps_in"
                )
        # Save stuff
        self.irreps_in = irreps_in
        # The output irreps of any graph module are whatever inputs it has, overwritten with whatever outputs it has.
        new_out = irreps_in.copy()
        new_out.update(irreps_out)
        self.irreps_out = new_out

    def _make_tracing_inputs(self, n):
        # We impliment this to be able to trace graph modules
        out = []
        for _ in range(n):
            batch = random.randint(1, 4)
            # TODO: handle None case
            # TODO: do only required inputs
            # TODO: dummy input if empty?
            out.append(
                {
                    "forward": (
                        {k: i.randn(batch, -1) for k, i in self.irreps_in.items()},
                    )
                }
            )
        return out


class SequentialGraphNetwork(GraphModuleMixin, torch.nn.Sequential):
    def __init__(
        self,
        modules: Union[Sequence[GraphModuleMixin], Dict[str, GraphModuleMixin]],
        init_args: Optional[list] = [],
    ):
        if isinstance(modules, dict):
            module_list = list(modules.values())
        else:
            module_list = list(modules)
        # check in/out irreps compatible
        for m1, m2 in zip(module_list, module_list[1:]):
            assert AtomicDataDict._irreps_compatible(m1.irreps_out, m2.irreps_in)
        self._init_irreps(
            irreps_in=module_list[0].irreps_in,
            my_irreps_in=module_list[0].irreps_in,
            irreps_out=module_list[-1].irreps_out,
        )
        if isinstance(modules, dict):
            if not isinstance(modules, OrderedDict):
                modules = OrderedDict(modules)
            # torch.nn.Sequential will name children correctly if passed an OrderedDict
            super().__init__(modules)
        else:
            super().__init__(*module_list)
        self.init_args = deepcopy(init_args)

    @classmethod
    def from_parameters(
        cls,
        shared_params: Mapping,
        layers: Dict[str, Union[Callable, Tuple[Callable, Dict[str, Any]]]],
    ):
        """construct the network from parameters

        Args:

        shared_params (dict): the parameters that are shared among all modules, can be overridden by the args
        args: list of modules

        Each module can be declared in three ways

            * callable constructor
            * callable constructor, prefix
            * callable constructor, prefix, kwargs

        Instantiate use four groups of parameters to initialize the modeul:

        1. the ones in shared_params that matches the callalble initialization arguments.
        2. the ones in shared_params that prefix_+(init argument)
        3. The ones in kwargs that matches (arg) or prefix_(arg)
        4. irreps_out from the previous model is taken as irreps_in

        """
        # note that dictionary ordered gueranteed in >=3.7, so its fine to do an ordered sequential as a dict.
        built_modules = []
        for name, builder in layers.items():
            if not isinstance(name, str):
                raise ValueError(f"`'name'` must be a str; got `{name}`")
            if isinstance(builder, tuple):
                builder, params = builder
            else:
                params = {}
            if not callable(builder):
                raise TypeError(
                    f"The builder has to be a class or a function. got {type(builder)}"
                )

            instance, _ = instantiate(
                builder=builder,
                prefix=name,
                positional_args=(
                    dict(irreps_in=built_modules[-1].irreps_out)
                    if len(built_modules) > 0
                    else {}
                ),
                optional_args=params,
                all_args=shared_params,
            )

            built_modules.append(instance)

        return cls(
            OrderedDict(zip(layers.keys(), built_modules)),
            init_args=[shared_params, layers],
        )

    # Copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#Sequential
    # with type annotations added
    def forward(self, input: AtomicDataDict.Type) -> AtomicDataDict.Type:
        for module in self:
            input = module(input)
        return input
