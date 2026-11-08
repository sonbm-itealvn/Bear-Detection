from det.sim_tools import _match_outputs

def test_match_outputs_by_producer_name():
    bitinfo = {
        "Gemm_output_Y": {"is_network_out": True, "ori_graph_name": "/classifier/classifier.1/Gemm"},
        "Conv_/head/heat/Conv_output_0_Y": {"is_network_out": True, "ori_graph_name": "/head/heat/Conv"},
        "Conv_/head/box/Conv_output_0_Y": {"is_network_out": True, "ori_graph_name": "/head/box/Conv"},
        "Conv_/stem/conv/Conv_output_0_Y": {"is_network_out": False, "ori_graph_name": "/stem/conv/Conv"},
    }
    m = _match_outputs(bitinfo)
    assert m == {"output": "Gemm_output_Y", "heat": "Conv_/head/heat/Conv_output_0_Y",
                 "box": "Conv_/head/box/Conv_output_0_Y"}


def test_dump_filename_replaces_slashes():
    from det.sim_tools import dump_name
    assert dump_name("Conv_/head/heat/Conv_output_0_Y") == "Conv__head_heat_Conv_output_0_Y.bin"
    assert dump_name("Gemm_output_Y") == "Gemm_output_Y.bin"
