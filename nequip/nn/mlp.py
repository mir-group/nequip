from math import sqrt
import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin

from typing import Optional


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
        )
        self.irreps_out[self.out_field] = o3.Irreps(
            [(self.mlp_module.dims[-1], (0, 1))]
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.out_field] = self.mlp_module(data[self.field])
        return data


class ScalarMLPFunction(torch.nn.Module):
    """Module implementing an MLP according to provided options.

    ``input_dim`` and ``output_dim`` are mandatory arguments.
    If only ``input_dim`` and ``output_dim`` are specified, this module defaults to a linear layer (corresponding to the default of ``hidden_layers_depth=0``).
    If ``hidden_layers_depth!=0``,  ``hidden_layers_width`` must be configured (an error will be raised if the default of ``hidden_layers_width=None`` is used).

    Args:
        nonlinearity (str): ``silu`` (default), ``mish``, ``gelu``, or ``None``
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
        nonlinearity_module = {
            None: torch.nn.Identity,
            "silu": torch.nn.SiLU,
            "mish": torch.nn.Mish,
            "gelu": torch.nn.GELU,
        }[nonlinearity]
        self.is_nonlinear = False  # updated below in loop

        # === build the MLP + weight init ===
        self.mlp = torch.nn.Sequential()
        for layer, (h_in, h_out) in enumerate(zip(self.dims, self.dims[1:])):

            # === weight initialization ===
            # normalize to preserve variance of forward activations or backward derivatives
            # we use "relu" gain (sqrt(2)) as a stand-in for the smooth nonlinearities we use, and only apply them if there is a nonlinearity
            # for forward (backward) norm, we don't include the gain for the first (last) layer
            # see https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
            if forward_weight_init:

                norm_dim = h_in
                gain = (
                    1.0
                    if isinstance(nonlinearity_module, torch.nn.Identity)
                    or (layer == 0)
                    else sqrt(2)
                )
            else:
                norm_dim = h_out
                gain = (
                    1.0
                    if isinstance(nonlinearity_module, torch.nn.Identity)
                    or (layer == self.num_layers - 1)
                    else sqrt(2)
                )
            self.mlp.append(ScalarNormalize(gain / sqrt(norm_dim)))
            del gain, norm_dim

            # === instantiate `Linear` ===
            linear_layer = torch.nn.Linear(h_in, h_out, bias=bias)
            torch.nn.init.uniform_(linear_layer.weight, -sqrt(3), sqrt(3))
            self.mlp.append(linear_layer)

            # === add nonlinearity (if any) except for last layer ===
            if layer != self.num_layers - 1:
                self.mlp.append(nonlinearity_module())
                # only update `self.is_nonlinear` when a nonlinearity is applied
                if not self.is_nonlinear:
                    self.is_nonlinear = not isinstance(
                        nonlinearity_module, torch.nn.Identity
                    )

    def forward(self, x):
        return self.mlp(x)


class ScalarNormalize(torch.nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, self.alpha)
