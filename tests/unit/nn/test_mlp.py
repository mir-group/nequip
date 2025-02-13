import torch
import e3nn
from nequip.nn.mlp import ScalarMLPFunction, DeepLinearMLP
from nequip.utils import torch_default_dtype, dtype_from_name
import pytest


@pytest.mark.parametrize("batch", [3, 5, 13])
@pytest.mark.parametrize("input_dim", [3, 7, 17])
@pytest.mark.parametrize("output_dim", [5, 11, 19])
@pytest.mark.parametrize("hl_depth", [0, 1, 2, 3])
@pytest.mark.parametrize("hl_width", [2, 5, 11])
@pytest.mark.parametrize("act", [None, "silu", "mish", "gelu"])
@pytest.mark.parametrize("bias", [True, False])
# @pytest.mark.parametrize("model_dtype", ["float32", "float64"])
def test_mlp(
    batch,
    input_dim,
    output_dim,
    hl_depth,
    hl_width,
    act,
    bias,
    model_dtype,
):

    tol = {
        "float32": 1e-6,
        "float64": 1e-12,
    }[model_dtype]

    compare_with_e3nn = (act is None or hl_depth == 0) and not bias

    with torch_default_dtype(dtype_from_name(model_dtype)):
        data = torch.randn(batch, input_dim)

        mlp_module = ScalarMLPFunction(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers_depth=hl_depth,
            hidden_layers_width=hl_width,
            nonlinearity=act,
            bias=bias,
        )

        if compare_with_e3nn:

            assert not mlp_module.is_nonlinear
            print(mlp_module.mlp)

            # in this regime, we can make direct comparisons to e3nn's implementation
            e3nn_mlp = e3nn.nn.FullyConnectedNet(
                hs=[input_dim] + (hl_depth * [hl_width]) + [output_dim],
                act=None,
            )
            # we need to make the weights consistent for testing
            # the weight setting logic depends on whether it's a single linear layer or a deep linear network
            if isinstance(mlp_module.mlp, DeepLinearMLP):
                with torch.no_grad():
                    for i, layer in enumerate(e3nn_mlp):
                        mlp_module.mlp.weights[i].copy_(layer.weight)
            else:
                with torch.no_grad():
                    mlp_module.mlp[0].weight.copy_(e3nn_mlp[0].weight)

    out = mlp_module(data)
    assert out.shape == (batch, output_dim)

    if compare_with_e3nn:

        e3nn_out = e3nn_mlp(data)
        assert torch.allclose(e3nn_out, out, rtol=tol, atol=tol), torch.max(
            torch.abs(e3nn_out - out)
        )
