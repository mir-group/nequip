# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

from ._tp_scatter_base import TensorProductScatter


def nequip_tp_desc(
    irreps1,
    irreps2,
    irreps3,
):
    """Construct the NequIP version of channelwise tensor product descriptor.

    subscripts: ``weights[uv],lhs[iu],rhs[jv],output[ku]``

    Args:
        irreps1 (Irreps): Irreps of the first operand.
        irreps2 (Irreps): Irreps of the second operand.
        irreps3 (Irreps): Irreps of the output to consider.
    """
    import cuequivariance as cue
    from cuequivariance.group_theory.irreps_array.irrep_utils import into_list_of_irrep
    import itertools

    # modified from `channelwise_tensor_product`
    # https://github.com/NVIDIA/cuEquivariance/blob/7236768147394a7da6abd7d5209d274704057eed/cuequivariance/cuequivariance/group_theory/descriptors/irreps_tp.py#L149

    G = irreps1.irrep_class
    irreps3_filter = into_list_of_irrep(G, irreps3)

    d = cue.SegmentedTensorProduct.from_subscripts("uv,iu,jv,kuv+ijk")

    for mul, ir in irreps1:
        d.add_segment(1, (ir.dim, mul))
    for mul, ir in irreps2:
        d.add_segment(2, (ir.dim, mul))

    irreps3 = []
    for (i1, (mul1, ir1)), (i2, (mul2, ir2)) in itertools.product(
        enumerate(irreps1), enumerate(irreps2)
    ):
        for ir3 in ir1 * ir2:
            if ir3 not in irreps3_filter:
                continue

            for cg in cue.clebsch_gordan(ir1, ir2, ir3):
                d.add_path(None, i1, i2, None, c=cg, dims={"u": mul1, "v": mul2})

                irreps3.append((mul1 * mul2, ir3))

    irreps3 = cue.Irreps(G, irreps3)
    irreps3, perm, inv = irreps3.sort()
    d = d.permute_segments(3, inv)
    d = d.normalize_paths_for_operand(-1)

    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps1.new_scalars(d.operands[0].size), cue.ir_mul),
            cue.IrrepsAndLayout(irreps1, cue.ir_mul),
            cue.IrrepsAndLayout(irreps2, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(irreps3, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


class CuEquivarianceTensorProductScatter(TensorProductScatter):
    def __init__(
        self,
        feature_irreps_in,
        irreps_edge_attr,
        irreps_mid,
        instructions,
    ) -> None:
        super().__init__(
            feature_irreps_in=feature_irreps_in,
            irreps_edge_attr=irreps_edge_attr,
            irreps_mid=irreps_mid,
            instructions=instructions,
        )
        # ^ we ensure that the base class keeps around a `self.tp` that carries its own set of persistent buffers
        # even though `self.tp` is not used, having its (persistent) buffers always around ensures state dict compatibility when adding on or removing this subclass module

        # === CuEq ===

        # we do lazy imports of cuequivariance to allow `nequip-package` to pick this file up even if cuequivariance is not installed
        # since `nequip-package` ignores files if it errors on loading the file

        import cuequivariance as cue
        import cuequivariance_torch as cuet
        from cuequivariance.group_theory.experimental.e3nn import O3_e3nn

        self.tp_conv = cuet.SegmentedPolynomial(
            nequip_tp_desc(
                cue.Irreps(O3_e3nn, feature_irreps_in),
                cue.Irreps(O3_e3nn, irreps_edge_attr),
                cue.Irreps(O3_e3nn, irreps_mid),
            )
            .flatten_coefficient_modes()
            .squeeze_modes()
            .polynomial,
            method="fused_tp",
            math_dtype=self.model_dtype,
        )

        self.transpose_feat = cuet.TransposeIrrepsLayout(
            feature_irreps_in, source=cue.mul_ir, target=cue.ir_mul
        )
        self.transpose_out = cuet.TransposeIrrepsLayout(
            irreps_mid, source=cue.ir_mul, target=cue.mul_ir
        )

    def forward(self, x, edge_attr, edge_weight, edge_dst, edge_src):
        return self.transpose_out(
            self.tp_conv(
                [edge_weight, self.transpose_feat(x), edge_attr],
                {1: edge_src},
                {0: x},
                {0: edge_dst},
            )[0]
        )
