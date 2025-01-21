import torch
from nequip.nn import ScalarMLPFunction
from nequip.utils import torch_default_dtype
import pytest


@pytest.mark.parametrize("batch", [3, 5, 13])
@pytest.mark.parametrize("input_dim", [3, 7, 17])
@pytest.mark.parametrize("output_dim", [5, 11, 19])
@pytest.mark.parametrize("hl_depth", [0, 1, 2, 3])
@pytest.mark.parametrize("hl_width", [2, 5, 11])
@pytest.mark.parametrize("act", [None, "silu", "mish", "gelu"])
@pytest.mark.parametrize("model_dtype", [torch.float32, torch.float64])
def test_mlp(
    batch,
    input_dim,
    output_dim,
    hl_depth,
    hl_width,
    act,
    model_dtype,
):
    with torch_default_dtype(model_dtype):
        data = torch.randn(batch, input_dim)

        mlp = ScalarMLPFunction(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers_depth=hl_depth,
            hidden_layers_width=hl_width,
            nonlinearity=act,
        )

    out = mlp(data)
    assert out.shape == (batch, output_dim)
