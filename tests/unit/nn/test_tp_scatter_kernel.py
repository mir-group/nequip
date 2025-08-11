import pytest
import torch
from e3nn import o3

from nequip.utils import torch_default_dtype, dtype_from_name
from nequip.utils.versions.torch_versions import _TORCH_GE_2_4
from nequip.nn._tp_scatter_base import TensorProductScatter


_CUEQ_INSTALLED = False
_OEQ_INSTALLED = False

# test data dimensions
NUM_NODES = 8
NUM_EDGES = 15

if _TORCH_GE_2_4:
    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_torch  # noqa: F401

        _CUEQ_INSTALLED = True
    except ImportError:
        pass

    try:
        import openequivariance  # noqa: F401

        _OEQ_INSTALLED = True
    except ImportError:
        pass


@pytest.mark.parametrize(
    "kernel_type",
    (["cueq"] if _CUEQ_INSTALLED else []) + (["oeq"] if _OEQ_INSTALLED else []),
)
@pytest.mark.parametrize(
    "feature_irreps_in",
    [
        "4x0e + 3x1o + 2x2e",
        "2x0e + 2x1o + 2x2e",
        "8x0e + 8x2e + 8x1o",
    ],
)
@pytest.mark.parametrize("irreps_edge_attr", ["0e + 1o", "0e + 1o + 2e"])
@pytest.mark.parametrize(
    "irreps_mid",
    [
        "0e + 1o + 2e",
        "2x0e + 2x1o + 2x2e",
        "24x0e + 32x1o + 16x1e + 16x2o + 32x2e",
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="kernel tests only work with GPUs"
)
def test_tp_scatter_kernel(
    kernel_type,
    feature_irreps_in,
    irreps_edge_attr,
    irreps_mid,
    dtype,
):
    """Test that kernel-accelerated TensorProductScatter produces identical results to the base implementation."""
    assert torch.cuda.is_available()
    device = "cuda"

    feature_irreps_in = o3.Irreps(feature_irreps_in)
    irreps_edge_attr = o3.Irreps(irreps_edge_attr)
    irreps_mid = o3.Irreps(irreps_mid)

    # build instructions exactly like InteractionBlock
    irreps_mid_list = []
    instructions = []

    for i, (mul, ir_in) in enumerate(feature_irreps_in):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_mid:
                    k = len(irreps_mid_list)
                    irreps_mid_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))

    if not instructions:
        pytest.skip("No valid tensor product instructions generated")

    # sort irreps and permute instructions like InteractionBlock
    irreps_mid = o3.Irreps(irreps_mid_list)
    irreps_mid, p, _ = irreps_mid.sort()

    # permute the output indexes to match sorted irreps
    instructions = [
        (i_in1, i_in2, p[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    with torch_default_dtype(dtype_from_name(dtype)):
        # create base TensorProductScatter
        tp_base = TensorProductScatter(
            feature_irreps_in=feature_irreps_in,
            irreps_edge_attr=irreps_edge_attr,
            irreps_mid=irreps_mid,
            instructions=instructions,
        ).to(device=device)

        # create kernel-accelerated version
        if kernel_type == "cueq":
            from nequip.nn._tp_scatter_cueq import CuEquivarianceTensorProductScatter

            tp_kernel = CuEquivarianceTensorProductScatter(
                feature_irreps_in=feature_irreps_in,
                irreps_edge_attr=irreps_edge_attr,
                irreps_mid=irreps_mid,
                instructions=instructions,
            ).to(device=device)
        elif kernel_type == "oeq":
            from nequip.nn._tp_scatter_oeq import OpenEquivarianceTensorProductScatter

            tp_kernel = OpenEquivarianceTensorProductScatter(
                feature_irreps_in=feature_irreps_in,
                irreps_edge_attr=irreps_edge_attr,
                irreps_mid=irreps_mid,
                instructions=instructions,
                use_opaque=False,
            ).to(device=device)

        tp_kernel.eval()

        # copy weights from base to kernel version
        with torch.no_grad():
            if hasattr(tp_kernel, "tp") and hasattr(tp_base, "tp"):
                # ensure weights are compatible
                if tp_kernel.tp.weight_numel == tp_base.tp.weight_numel:
                    tp_kernel.tp.weight.copy_(tp_base.tp.weight)

        # set up test data

        # create random input data
        x = feature_irreps_in.randn(NUM_NODES, -1, device=device)
        edge_attr = irreps_edge_attr.randn(NUM_EDGES, -1, device=device)
        edge_weight = torch.randn(NUM_EDGES, tp_base.tp.weight_numel, device=device)
        edge_src = torch.randint(
            0, NUM_NODES, (NUM_EDGES,), dtype=torch.long, device=device
        )
        edge_dst = torch.randint(
            0, NUM_NODES, (NUM_EDGES,), dtype=torch.long, device=device
        )

        # test tolerance based on dtype
        tol = {torch.float32: 1e-5, torch.float64: 1e-10}[torch.get_default_dtype()]

        # test forward pass
        with torch.no_grad():
            out_base = tp_base(x, edge_attr, edge_weight, edge_dst, edge_src)
            out_kernel = tp_kernel(x, edge_attr, edge_weight, edge_dst, edge_src)
            torch.testing.assert_close(out_base, out_kernel, atol=tol, rtol=tol)

        # test gradients
        for param_name in ["x", "edge_attr", "edge_weight"]:
            inputs = {"x": x, "edge_attr": edge_attr, "edge_weight": edge_weight}
            inputs[param_name].requires_grad_(True)

            out_base = tp_base(**inputs, edge_dst=edge_dst, edge_src=edge_src)
            out_kernel = tp_kernel(**inputs, edge_dst=edge_dst, edge_src=edge_src)

            grad_output = torch.randn_like(out_base)

            grad_base = torch.autograd.grad(
                out_base, inputs[param_name], grad_output, retain_graph=True
            )[0]
            grad_kernel = torch.autograd.grad(
                out_kernel, inputs[param_name], grad_output, retain_graph=True
            )[0]

            torch.testing.assert_close(grad_base, grad_kernel, atol=tol, rtol=tol)

            inputs[param_name].requires_grad_(False)
