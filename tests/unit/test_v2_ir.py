"""
Unit tests for ForgeTune v2.0 IR, JSON Serialization, and Graph Verifier.
"""

from forgetune import _forgetune_cpp as cpp
from forgetune.graph_adapter import Node, Graph

def test_v2_json_roundtrip_with_conv2d():
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, 3, 16, 16]))
    w = g.add_node(Node(1, "CONST", shape=[8, 3, 3, 3], const_data=[0.01] * (8 * 3 * 3 * 3)))
    b = g.add_node(Node(2, "CONST", shape=[8], const_data=[0.1] * 8))
    conv = g.add_node(Node(3, "CONV2D", inputs=[x, w, b], shape=[-1, 8, 16, 16],
                           conv_attrs={"stride_h": 1, "stride_w": 1, "pad_h": 1, "pad_w": 1,
                                       "dilation_h": 1, "dilation_w": 1, "groups": 1}))
    relu = g.add_node(Node(4, "RELU", inputs=[conv], shape=[-1, 8, 16, 16]))
    flat = g.add_node(Node(5, "FLATTEN", inputs=[relu], shape=[-1, 8 * 16 * 16]))
    g.set_inputs([x])
    g.set_outputs([flat])

    json_str = g.to_json()
    assert '"schema_version_major": 2' in json_str
    assert '"conv_attrs"' in json_str

    g_deser = Graph.from_json(json_str)
    assert g_deser.inputs == [0]
    assert g_deser.outputs == [5]
    assert g_deser.nodes[3].op == "CONV2D"
    assert g_deser.nodes[3].conv_attrs["stride_h"] == 1
    assert g_deser.nodes[3].conv_attrs["pad_h"] == 1
    assert g_deser.nodes[5].op == "FLATTEN"
    assert g_deser.nodes[5].shape == [-1, 8 * 16 * 16]

    # Verify with C++ parser as well
    cpp_graph = cpp.json_to_graph(json_str)
    res = cpp.verify_graph(cpp_graph, False)
    assert res.passed, f"C++ graph verification failed: {res.to_string()}"

def test_v2_new_ops_roundtrip_and_verify():
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, 16]))
    w = g.add_node(Node(1, "CONST", shape=[16, 16], const_data=[0.05] * 256))
    mm = g.add_node(Node(2, "MATMUL", inputs=[x, w], shape=[-1, 16]))
    sig = g.add_node(Node(3, "SIGMOID", inputs=[mm], shape=[-1, 16]))
    tnh = g.add_node(Node(4, "TANH", inputs=[sig], shape=[-1, 16]))
    mul = g.add_node(Node(5, "MUL", inputs=[mm, tnh], shape=[-1, 16]))
    g.set_inputs([x])
    g.set_outputs([mul])

    json_str = g.to_json()
    cpp_graph = cpp.json_to_graph(json_str)
    res = cpp.verify_graph(cpp_graph, False)
    assert res.passed, f"C++ graph verification failed: {res.to_string()}"

def test_verifier_rejects_non_leading_dynamic_dimension():
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[4, -1]))
    g.set_inputs([x])
    g.set_outputs([x])

    json_str = g.to_json()
    cpp_graph = cpp.json_to_graph(json_str)
    res = cpp.verify_graph(cpp_graph, False)
    assert not res.passed
    assert any("Only leading batch dimension (dim 0) may be dynamic" in err for err in res.errors)

def test_verifier_rejects_const_with_dynamic_dimension():
    g = Graph()
    c = g.add_node(Node(0, "CONST", shape=[-1, 4], const_data=[1.0] * 8))
    g.set_inputs([])
    g.set_outputs([c])

    json_str = g.to_json()
    cpp_graph = cpp.json_to_graph(json_str)
    res = cpp.verify_graph(cpp_graph, False)
    assert not res.passed
    assert any("CONST node" in err and "cannot have dynamic dimensions" in err for err in res.errors)

def test_verifier_rejects_conv2d_channel_mismatch():
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, 3, 16, 16]))
    # Weight expects 4 channels, but input has 3
    w = g.add_node(Node(1, "CONST", shape=[8, 4, 3, 3], const_data=[0.01] * (8 * 4 * 3 * 3)))
    conv = g.add_node(Node(2, "CONV2D", inputs=[x, w], shape=[-1, 8, 14, 14],
                           conv_attrs={"stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0,
                                       "dilation_h": 1, "dilation_w": 1, "groups": 1}))
    g.set_inputs([x])
    g.set_outputs([conv])

    json_str = g.to_json()
    cpp_graph = cpp.json_to_graph(json_str)
    res = cpp.verify_graph(cpp_graph, False)
    assert not res.passed
    assert any("channel mismatch" in err for err in res.errors)

def test_verifier_rejects_conv2d_dynamic_weight():
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, 3, 16, 16]))
    w = g.add_node(Node(1, "CONST", shape=[-1, 3, 3, 3], const_data=[0.01] * (8 * 3 * 3 * 3)))
    conv = g.add_node(Node(2, "CONV2D", inputs=[x, w], shape=[-1, 8, 14, 14],
                           conv_attrs={"stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0,
                                       "dilation_h": 1, "dilation_w": 1, "groups": 1}))
    g.set_inputs([x])
    g.set_outputs([conv])

    json_str = g.to_json()
    cpp_graph = cpp.json_to_graph(json_str)
    res = cpp.verify_graph(cpp_graph, False)
    assert not res.passed
    assert any("weight W cannot have dynamic dimensions" in err or "CONST node" in err for err in res.errors)

def test_verifier_rejects_flatten_invalid_rank():
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, 3, 16]))  # Rank-3
    flat = g.add_node(Node(1, "FLATTEN", inputs=[x], shape=[-1, 48]))
    g.set_inputs([x])
    g.set_outputs([flat])

    json_str = g.to_json()
    cpp_graph = cpp.json_to_graph(json_str)
    res = cpp.verify_graph(cpp_graph, False)
    assert not res.passed
    assert any("requires rank-2 or rank-4 input" in err for err in res.errors)
