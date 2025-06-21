from nequip.data import AtomicDataDict
from nequip.nn import (
    GraphModel,
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
)
from nequip.nn.compile import CompileGraphModel
from nequip.nn.embedding import NodeTypeEmbed
from nequip.utils import dtype_from_name, torch_default_dtype


def test_graph_model_flags(model_dtype, CH3CHO):
    """Test that GraphModel and CompileGraphModel have correct identification flags."""
    mdtype = dtype_from_name(model_dtype)

    with torch_default_dtype(mdtype):
        # create model
        embed = NodeTypeEmbed(type_names=["C", "O", "H"], num_features=8)
        linear = AtomwiseLinear(
            irreps_in=embed.irreps_out,
            irreps_out="1x0e",
            out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        )
        reduce = AtomwiseReduce(
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
            irreps_in=linear.irreps_out,
        )
        model = SequentialGraphNetwork(
            {
                "embed": embed,
                "linear": linear,
                "reduce": reduce,
            }
        )
        model_config = {
            "model_dtype": model_dtype,
            "type_names": ["C", "O", "H"],
            "r_max": 2.0,
        }

        # test GraphModel
        graph_model = GraphModel(model, model_config=model_config)
        assert hasattr(graph_model, "is_graph_model")
        assert graph_model.is_graph_model is True
        assert hasattr(graph_model, "is_compile_graph_model")
        assert graph_model.is_compile_graph_model is False

        # test CompileGraphModel
        compile_graph_model = CompileGraphModel(model, model_config=model_config)
        assert hasattr(compile_graph_model, "is_graph_model")
        assert compile_graph_model.is_graph_model is True
        assert hasattr(compile_graph_model, "is_compile_graph_model")
        assert compile_graph_model.is_compile_graph_model is True
