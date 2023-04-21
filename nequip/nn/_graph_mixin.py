import random
from typing import Dict, Tuple, Callable, Any, Sequence, Union, Mapping, Optional
from collections import OrderedDict

import torch

from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.utils import instantiate


class GraphModuleMixin:
    r"""Mixin parent class for ``torch.nn.Module``s that act on and return ``AtomicDataDict.Type`` graph data.

    All such classes should call ``_init_irreps`` in their ``__init__`` functions with information on the data fields they expect, require, and produce, as well as their corresponding irreps.
    """

    def _init_irreps(
        self,
        irreps_in: Dict[str, Any] = {},
        my_irreps_in: Dict[str, Any] = {},
        required_irreps_in: Sequence[str] = [],
        irreps_out: Dict[str, Any] = {},
    ):
        """Setup the expected data fields and their irreps for this graph module.

        ``None`` is a valid irreps in the context for anything that is invariant but not well described by an ``e3nn.o3.Irreps``. An example are edge indexes in a graph, which are invariant but are integers, not ``0e`` scalars.

        Args:
            irreps_in (dict): maps names of all input fields from previous modules or
                data to their corresponding irreps
            my_irreps_in (dict): maps names of fields to the irreps they must have for
                this graph module. Will be checked for consistancy with ``irreps_in``
            required_irreps_in: sequence of names of fields that must be present in
                ``irreps_in``, but that can have any irreps.
            irreps_out (dict): mapping names of fields that are modified/output by
                this graph module to their irreps.
        """
        # Coerce
        irreps_in = {} if irreps_in is None else irreps_in
        irreps_in = AtomicDataDict._fix_irreps_dict(irreps_in)
        # positions are *always* 1o, and always present
        if AtomicDataDict.POSITIONS_KEY in irreps_in:
            if irreps_in[AtomicDataDict.POSITIONS_KEY] != o3.Irreps("1x1o"):
                raise ValueError(
                    f"Positions must have irreps 1o, got instead `{irreps_in[AtomicDataDict.POSITIONS_KEY]}`"
                )
        irreps_in[AtomicDataDict.POSITIONS_KEY] = o3.Irreps("1o")
        # edges are also always present
        if AtomicDataDict.EDGE_INDEX_KEY in irreps_in:
            if irreps_in[AtomicDataDict.EDGE_INDEX_KEY] is not None:
                raise ValueError(
                    f"Edge indexes must have irreps None, got instead `{irreps_in[AtomicDataDict.EDGE_INDEX_KEY]}`"
                )
        irreps_in[AtomicDataDict.EDGE_INDEX_KEY] = None

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

    def _add_independent_irreps(self, irreps: Dict[str, Any]):
        """
        Insert some independent irreps that need to be exposed to the self.irreps_in and self.irreps_out.
        The terms that have already appeared in the irreps_in will be removed.

        Args:
            irreps (dict): maps names of all new fields
        """

        irreps = {
            key: irrep for key, irrep in irreps.items() if key not in self.irreps_in
        }
        irreps_in = AtomicDataDict._fix_irreps_dict(irreps)
        irreps_out = AtomicDataDict._fix_irreps_dict(
            {key: irrep for key, irrep in irreps.items() if key not in self.irreps_out}
        )
        self.irreps_in.update(irreps_in)
        self.irreps_out.update(irreps_out)

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
                        {
                            k: i.randn(batch, -1)
                            for k, i in self.irreps_in.items()
                            if i is not None
                        },
                    )
                }
            )
        return out


class SequentialGraphNetwork(GraphModuleMixin, torch.nn.Sequential):
    r"""A ``torch.nn.Sequential`` of ``GraphModuleMixin``s.

    Args:
        modules (list or dict of ``GraphModuleMixin``s): the sequence of graph modules. If a list, the modules will be named ``"module0", "module1", ...``.
    """

    def __init__(
        self,
        modules: Union[Sequence[GraphModuleMixin], Dict[str, GraphModuleMixin]],
    ):
        if isinstance(modules, dict):
            module_list = list(modules.values())
        else:
            module_list = list(modules)
        # check in/out irreps compatible
        for m1, m2 in zip(module_list, module_list[1:]):
            assert AtomicDataDict._irreps_compatible(
                m1.irreps_out, m2.irreps_in
            ), f"Incompatible irreps_out from {type(m1).__name__} for input to {type(m2).__name__}: {m1.irreps_out} -> {m2.irreps_in}"
        self._init_irreps(
            irreps_in=module_list[0].irreps_in,
            my_irreps_in=module_list[0].irreps_in,
            irreps_out=module_list[-1].irreps_out,
        )
        # torch.nn.Sequential will name children correctly if passed an OrderedDict
        if isinstance(modules, dict):
            modules = OrderedDict(modules)
        else:
            modules = OrderedDict((f"module{i}", m) for i, m in enumerate(module_list))
        super().__init__(modules)

    @classmethod
    def from_parameters(
        cls,
        shared_params: Mapping,
        layers: Dict[str, Union[Callable, Tuple[Callable, Dict[str, Any]]]],
        irreps_in: Optional[dict] = None,
    ):
        r"""Construct a ``SequentialGraphModule`` of modules built from a shared set of parameters.

        For some layer, a parameter with name ``param`` will be taken, in order of priority, from:
          1. The specific value in the parameter dictionary for that layer, if provided
          2. ``name_param`` in ``shared_params`` where ``name`` is the name of the layer
          3. ``param`` in ``shared_params``

        Args:
            shared_params (dict-like): shared parameters from which to pull when instantiating the module
            layers (dict): dictionary mapping unique names of layers to either:
                  1. A callable (such as a class or function) that can be used to ``instantiate`` a module for that layer
                  2. A tuple of such a callable and a dictionary mapping parameter names to values. The given dictionary of parameters will override for this layer values found in ``shared_params``.
                Options 1. and 2. can be mixed.
            irreps_in (optional dict): ``irreps_in`` for the first module in the sequence.

        Returns:
            The constructed SequentialGraphNetwork.
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
                    dict(
                        irreps_in=(
                            built_modules[-1].irreps_out
                            if len(built_modules) > 0
                            else irreps_in
                        )
                    )
                ),
                optional_args=params,
                all_args=shared_params,
            )

            if not isinstance(instance, GraphModuleMixin):
                raise TypeError(
                    f"Builder `{builder}` for layer with name `{name}` did not return a GraphModuleMixin, instead got a {type(instance).__name__}"
                )

            built_modules.append(instance)

        return cls(
            OrderedDict(zip(layers.keys(), built_modules)),
        )

    @torch.jit.unused
    def append(self, name: str, module: GraphModuleMixin) -> None:
        r"""Append a module to the SequentialGraphNetwork.

        Args:
            name (str): the name for the module
            module (GraphModuleMixin): the module to append
        """
        assert AtomicDataDict._irreps_compatible(self.irreps_out, module.irreps_in)
        self.add_module(name, module)
        self.irreps_out = dict(module.irreps_out)
        return

    @torch.jit.unused
    def append_from_parameters(
        self,
        shared_params: Mapping,
        name: str,
        builder: Callable,
        params: Dict[str, Any] = {},
    ) -> GraphModuleMixin:
        r"""Build a module from parameters and append it.

        Args:
            shared_params (dict-like): shared parameters from which to pull when instantiating the module
            name (str): the name for the module
            builder (callable): a class or function to build a module
            params (dict, optional): extra specific parameters for this module that take priority over those in ``shared_params``

        Returns:
            the build module
        """
        instance, _ = instantiate(
            builder=builder,
            prefix=name,
            positional_args=(dict(irreps_in=self[-1].irreps_out)),
            optional_args=params,
            all_args=shared_params,
        )
        self.append(name, instance)
        return instance

    @torch.jit.unused
    def insert(
        self,
        name: str,
        module: GraphModuleMixin,
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> None:
        """Insert a module after the module with name ``after``.

        Args:
            name: the name of the module to insert
            module: the moldule to insert
            after: the module to insert after
            before: the module to insert before
        """

        if (before is None) is (after is None):
            raise ValueError("Only one of before or after argument needs to be defined")
        elif before is None:
            insert_location = after
        else:
            insert_location = before

        # This checks names, etc.
        self.add_module(name, module)
        # Now insert in the right place by overwriting
        names = list(self._modules.keys())
        modules = list(self._modules.values())
        idx = names.index(insert_location)
        if before is None:
            idx += 1
        names.insert(idx, name)
        modules.insert(idx, module)

        self._modules = OrderedDict(zip(names, modules))

        module_list = list(self._modules.values())

        # sanity check the compatibility
        if idx > 0:
            assert AtomicDataDict._irreps_compatible(
                module_list[idx - 1].irreps_out, module.irreps_in
            )
        if len(module_list) > idx:
            assert AtomicDataDict._irreps_compatible(
                module_list[idx + 1].irreps_in, module.irreps_out
            )

        # insert the new irreps_out to the later modules
        for module_id, next_module in enumerate(module_list[idx + 1 :]):
            next_module._add_independent_irreps(module.irreps_out)

        # update the final wrapper irreps_out
        self.irreps_out = dict(module_list[-1].irreps_out)

        return

    @torch.jit.unused
    def insert_from_parameters(
        self,
        shared_params: Mapping,
        name: str,
        builder: Callable,
        params: Dict[str, Any] = {},
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> GraphModuleMixin:
        r"""Build a module from parameters and insert it after ``after``.

        Args:
            shared_params (dict-like): shared parameters from which to pull when instantiating the module
            name (str): the name for the module
            builder (callable): a class or function to build a module
            params (dict, optional): extra specific parameters for this module that take priority over those in ``shared_params``
            after: the name of the module to insert after
            before: the name of the module to insert before

        Returns:
            the inserted module
        """
        if (before is None) is (after is None):
            raise ValueError("Only one of before or after argument needs to be defined")
        elif before is None:
            insert_location = after
        else:
            insert_location = before
        idx = list(self._modules.keys()).index(insert_location) - 1
        if before is None:
            idx += 1
        instance, _ = instantiate(
            builder=builder,
            prefix=name,
            positional_args=(dict(irreps_in=self[idx].irreps_out)),
            optional_args=params,
            all_args=shared_params,
        )
        self.insert(after=after, before=before, name=name, module=instance)
        return instance

    # Copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#Sequential
    # with type annotations added
    def forward(self, input: AtomicDataDict.Type) -> AtomicDataDict.Type:
        for module in self:
            input = module(input)
        return input
