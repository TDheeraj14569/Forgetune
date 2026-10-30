"""
Unit tests for ForgeTune v2.0 Operator Fusion: CONV2D + RELU and legality gating.
"""

import numpy as np
from forgetune.graph_adapter import Node, Graph
from forgetune import _forgetune_cpp as cpp

def test_conv2d_relu_fusion_success():
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, 3, 16, 16]))
    w = g.add_node(Node(1, "CONST", shape=[8, 3, 3, 3], const_data=[0.01] * 216))
    b = g.add_node(Node(2, "CONST", shape=[8], const_data=[0.05] * 8))
    conv = g.add_node(Node(3, "CONV2D", inputs=[x, w, b], shape=[-1, 8, 16, 16],
                           conv_attrs={"stride_h": 1, "stride_w": 1, "pad_h": 1, "pad_w": 1,
                                       "dilation_h": 1, "dilation_w": 1, "groups": 1}))
    relu = g.add_node(Node(4, "RELU", inputs=[conv], shape=[-1, 8, 16, 16]))
    g.set_inputs([x])
    g.set_outputs([relu])

    cpp_graph = cpp.json_to_graph(g.to_json())
    initial_count = cpp_graph.live_node_count()
    assert initial_count == 5

    opt = cpp.Optimizer()
    report = opt.optimize(cpp_graph, enable_fusion=True)

    assert report.any_changed
    # CONV2D and RELU fused into 1 FUSED_CONV2D_RELU node
    assert cpp_graph.live_node_count() == 4

    # Verify that the fused node exists and has OpKind.FUSED_CONV2D_RELU
    fused_found = False
    for nid in cpp_graph.topological_order():
        node = cpp_graph.get_node(nid)
        if node.op == cpp.OpKind.FUSED_CONV2D_RELU:
            fused_found = True
            assert node.conv_attrs.stride_h == 1
            assert node.conv_attrs.pad_h == 1
            assert node.inputs == [x, w, b]
    assert fused_found

    # Verify numerical equivalence
    interp = cpp.Interpreter()
    input_data = np.random.randn(2, 3, 16, 16).astype(np.float32)

    unfused_graph = cpp.json_to_graph(g.to_json())
    out_unfused = interp.evaluate(unfused_graph, {0: input_data})[4]
    out_fused = interp.evaluate(cpp_graph, {0: input_data})[cpp_graph.get_outputs()[0]]

    np.testing.assert_allclose(out_fused, out_unfused, atol=1e-5, rtol=1e-4)

def test_conv2d_relu_fusion_rejected_when_multi_consumer():
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, 3, 16, 16]))
    w = g.add_node(Node(1, "CONST", shape=[8, 3, 3, 3], const_data=[0.01] * 216))
    conv = g.add_node(Node(2, "CONV2D", inputs=[x, w], shape=[-1, 8, 14, 14],
                           conv_attrs={"stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0,
                                       "dilation_h": 1, "dilation_w": 1, "groups": 1}))
    # Two consumers of conv: relu and sigmoid
    relu = g.add_node(Node(3, "RELU", inputs=[conv], shape=[-1, 8, 14, 14]))
    sig = g.add_node(Node(4, "SIGMOID", inputs=[conv], shape=[-1, 8, 14, 14]))
    add = g.add_node(Node(5, "ADD", inputs=[relu, sig], shape=[-1, 8, 14, 14]))
    g.set_inputs([x])
    g.set_outputs([add])

    cpp_graph = cpp.json_to_graph(g.to_json())
    cpp_graph.live_node_count()

    opt = cpp.Optimizer()
    opt.optimize(cpp_graph, enable_fusion=True)

    # Fusion must be rejected because CONV2D has multiple consumers
    for nid in cpp_graph.topological_order():
        node = cpp_graph.get_node(nid)
        assert node.op != cpp.OpKind.FUSED_CONV2D_RELU

def test_conv2d_relu_fusion_rejected_when_conv_is_graph_output():
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, 3, 16, 16]))
    w = g.add_node(Node(1, "CONST", shape=[8, 3, 3, 3], const_data=[0.01] * 216))
    conv = g.add_node(Node(2, "CONV2D", inputs=[x, w], shape=[-1, 8, 14, 14],
                           conv_attrs={"stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0,
                                       "dilation_h": 1, "dilation_w": 1, "groups": 1}))
    relu = g.add_node(Node(3, "RELU", inputs=[conv], shape=[-1, 8, 14, 14]))
    # conv also consumed by an extra node so it has multiple active consumers
    extra = g.add_node(Node(4, "ADD", inputs=[relu, conv], shape=[-1, 8, 14, 14]))
    g.set_inputs([x])
    g.set_outputs([extra])

    cpp_graph = cpp.json_to_graph(g.to_json())
    opt = cpp.Optimizer()
    opt.optimize(cpp_graph, enable_fusion=True)

    # Fusion must be rejected because conv is an exported output
    for nid in cpp_graph.topological_order():
        node = cpp_graph.get_node(nid)
        assert node.op != cpp.OpKind.FUSED_CONV2D_RELU
