# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
from math import sqrt, prod
import torch

from e3nn.o3._irreps import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin
from .nonlinearities import ShiftedSoftplus

from typing import Optional, Final, Dict


_NONLINEARITY_MAP: Final[Dict[str, torch.nn.Module]] = {
    # NOTE: we include str options for `None` so that the parser always works
    None: torch.nn.Identity,
    "None": torch.nn.Identity,
    "null": torch.nn.Identity,
    "silu": torch.nn.SiLU,
    "mish": torch.nn.Mish,
    "gelu": torch.nn.GELU,
    "ssp": ShiftedSoftplus,
    "tanh": torch.nn.Tanh,
}


@compile_mode("script")
class ScalarMLP(GraphModuleMixin, torch.nn.Module):
    """Apply an MLP to some scalar field."""

    field: str
    out_field: str

    def __init__(
        self,
        output_dim: int,
        hidden_layers_depth: int = 0,
        hidden_layers_width: Optional[int] = None,
        nonlinearity: Optional[str] = "silu",
        bias: bool = False,
        forward_weight_init: bool = True,
        init_mode: str = "uniform",
        parametrization: Optional[str] = None,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: Optional[str] = None,
        irreps_in=None,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field if out_field is not None else field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.field],
        )

        assert len(self.irreps_in[self.field]) == 1
        assert self.irreps_in[self.field][0].ir == (0, 1)  # scalars
        self.mlp_module = ScalarMLPFunction(
            input_dim=self.irreps_in[self.field][0].mul,
            output_dim=output_dim,
            hidden_layers_depth=hidden_layers_depth,
            hidden_layers_width=hidden_layers_width,
            nonlinearity=nonlinearity,
            bias=bias,
            forward_weight_init=forward_weight_init,
            init_mode=init_mode,
            parametrization=parametrization,
        )
        self.irreps_out[self.out_field] = Irreps([(self.mlp_module.dims[-1], (0, 1))])

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.out_field] = self.mlp_module(data[self.field])
        return data


@compile_mode("script")
class ScalarMLPFunction(torch.nn.Module):
    """Module implementing an MLP according to provided options.

    ``input_dim`` and ``output_dim`` are mandatory arguments.
    If only ``input_dim`` and ``output_dim`` are specified, this module defaults to a linear layer (corresponding to the default of ``hidden_layers_depth=0``).
    If ``hidden_layers_depth!=0``,  ``hidden_layers_width`` must be configured (an error will be raised if the default of ``hidden_layers_width=None`` is used).

    Args:
        nonlinearity (str): ``silu`` (default), ``mish``, ``gelu``, ``ssp``, ``tanh``, ``None``, ``null``, or ``"None"``
        bias (bool): whether a bias is included (default ``False``)
        forward_weight_init (bool): whether to initialize weights to preserve forward activation variance (default ``True``) or initialize weights to preserve backward gradient variance
    """

    num_layers: int
    bias: bool
    is_nonlinear: bool

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers_depth: int = 0,
        hidden_layers_width: Optional[int] = None,
        nonlinearity: Optional[str] = "silu",
        bias: bool = False,
        forward_weight_init: bool = True,
        init_mode: str = "uniform",
        parametrization: Optional[str] = None,
    ):
        super().__init__()
        self.bias = bias

        # === process MLP dims ===
        if hidden_layers_depth != 0:
            assert hidden_layers_depth > 0 and hidden_layers_width > 0
        hidden_layers_dims = hidden_layers_depth * [hidden_layers_width]
        self.dims = [input_dim] + hidden_layers_dims + [output_dim]
        self.num_layers = len(self.dims) - 1
        assert self.num_layers >= 1
        # NOTE: `input_dim` and `output_dim` are always mandatory, which default to at least a linear
        # a one-layer MLP is a linear layer

        # === handle nonlinearity ===
        # TODO: maybe adapt gain to be nonlinearity dependent
        if nonlinearity not in _NONLINEARITY_MAP:
            available_options = list(_NONLINEARITY_MAP.keys())
            raise ValueError(
                f"Unknown nonlinearity '{nonlinearity}'. Available options: {available_options}"
            )
        nonlinearity_module = _NONLINEARITY_MAP[nonlinearity]
        self.is_nonlinear = False  # updated below in loop

        # === build the MLP + weight init ===
        mlp = torch.nn.Sequential()
        for layer, (h_in, h_out) in enumerate(zip(self.dims, self.dims[1:])):
            # === weight initialization ===
            # normalize to preserve variance of forward activations or backward derivatives
            # we use "relu" gain (sqrt(2)) as a stand-in for the smooth nonlinearities we use, and only apply them if there is a nonlinearity
            # for forward (backward) norm, we don't include the gain for the first (last) layer
            # see https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
            if forward_weight_init:
                norm_dim = h_in
                gain = 1.0 if nonlinearity is None or (layer == 0) else sqrt(2)
            else:
                norm_dim = h_out
                gain = (
                    1.0
                    if nonlinearity is None or (layer == self.num_layers - 1)
                    else sqrt(2)
                )
            # === instantiate `Linear` ===
            linear_layer = ScalarLinearLayer(
                in_features=h_in,
                out_features=h_out,
                alpha=gain / sqrt(norm_dim),
                bias=bias,
                init_mode=init_mode,
            )

            # apply parametrization if specified
            if parametrization == "spectral_norm":
                torch.nn.utils.parametrizations.spectral_norm(
                    linear_layer, "weight", dim=1
                )
            elif parametrization == "weight_norm":
                torch.nn.utils.parametrizations.weight_norm(
                    linear_layer, "weight", dim=1
                )
            elif parametrization == "orthogonal":
                torch.nn.utils.parametrizations.orthogonal(linear_layer, "weight")
            elif parametrization is not None:
                raise ValueError(
                    f"Unknown parametrization '{parametrization}'. "
                    "Available options: None, 'weight_norm', 'orthogonal', 'spectral_norm'"
                )

            mlp.append(linear_layer)
            del gain, norm_dim

            # === add nonlinearity (if any) except for last layer ===
            if (layer != self.num_layers - 1) and (nonlinearity is not None):
                # only update `self.is_nonlinear` when a nonlinearity is applied
                mlp.append(nonlinearity_module())
                self.is_nonlinear = True

        # use `multidot` based implementation for deep linear net (no nonlinearity, no bias, more than one layer)
        # otherwise use the `mlp` built in init
        if (not self.is_nonlinear) and (not self.bias) and (self.num_layers > 1):
            self.mlp = DeepLinearMLP(mlp)
            del mlp
        else:
            self.mlp = mlp

    def forward(self, x):
        return self.mlp(x)


class DeepLinearMLP(torch.nn.Module):
    def __init__(self, mlp) -> None:
        super().__init__()
        self.weights = torch.nn.ParameterList()
        alphas = []
        for this_idx, mlp_idx in enumerate(range(len(mlp))):
            new_weight = torch.clone(mlp[mlp_idx].weight)
            self.weights.append(new_weight)
            del new_weight
            alphas.append(mlp[mlp_idx].alpha)
        alpha = prod(alphas)
        # the constant has to be a buffer for constant-folding to happen with `torch.compile(...dynamic=True)`
        # `persistent=False` for backwards compatibility of checkpoint files
        # (and technically preserves the old behavior when using a float in that it's also not persistent)
        # `alpha` is already a torch.Tensor here
        self.register_buffer("alpha", alpha, persistent=False)
        del alphas

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = torch.mul(
            torch.linalg.multi_dot([weight for weight in self.weights]), self.alpha
        )
        return torch.mm(input, weight)


class ScalarLinearLayer(torch.nn.Module):
    """Module implementing a linear layer with a scaling factor `alpha` applied to the weights."""

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 1.0,
        bias: bool = False,
        init_mode: str = "uniform",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # the constant has to be a buffer for constant-folding to happen with `torch.compile(...dynamic=True)`
        # `persistent=False` for backwards compatibility of checkpoint files
        # (and technically preserves the old behavior when using a float in that it's also not persistent)
        self.register_buffer("alpha", torch.tensor(alpha), persistent=False)
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        # initialize weights based on init_mode
        if init_mode == "uniform":
            # initialize weights to uniform distribution with mean 0 variance 1
            torch.nn.init.uniform_(self.weight, -sqrt(3), sqrt(3))
        elif init_mode == "normal":
            # initialize weights to normal distribution with mean 0 std 1
            torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)
        else:
            raise ValueError(
                f"Unknown init_mode: {init_mode}. Must be 'uniform' or 'normal'."
            )
        # initialize bias (if any) to zeros
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # compute scaled weights separately to be constant folded
        weight = self.weight * self.alpha
        if self.bias is None:
            return torch.mm(input, weight)
        else:
            return torch.addmm(self.bias, input, weight)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
