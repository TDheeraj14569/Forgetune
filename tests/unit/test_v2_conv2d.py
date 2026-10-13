"""
Unit tests for ForgeTune v2.0 CONV2D and FUSED_CONV2D_RELU.
Compares both Python and C++ reference interpreters against PyTorch ground truth.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from forgetune.graph_adapter import Node, Graph
from forgetune import _forgetune_cpp as cpp

TOL_ATOL = 1e-4
TOL_RTOL = 1e-4

def test_conv2d_3x3_stride1_pad1_with_bias():
    for B in [1, 3]:
        C_in, C_out = 3, 8
        H_in, W_in = 16, 16
        Kh, Kw = 3, 3

        torch.manual_seed(10 + B)
        t_x = torch.randn(B, C_in, H_in, W_in, dtype=torch.float32)
        t_w = torch.randn(C_out, C_in, Kh, Kw, dtype=torch.float32) * 0.1
        t_b = torch.randn(C_out, dtype=torch.float32) * 0.05

        t_out = F.conv2d(t_x, t_w, bias=t_b, stride=1, padding=1)

        g = Graph()
        x = g.add_node(Node(0, "INPUT", shape=[-1, C_in, H_in, W_in]))
        w = g.add_node(Node(1, "CONST", shape=[C_out, C_in, Kh, Kw], const_data=t_w.flatten().tolist()))
        b = g.add_node(Node(2, "CONST", shape=[C_out], const_data=t_b.tolist()))
        conv = g.add_node(Node(3, "CONV2D", inputs=[x, w, b], shape=[-1, C_out, 16, 16],
                               conv_attrs={"stride_h": 1, "stride_w": 1, "pad_h": 1, "pad_w": 1,
                                           "dilation_h": 1, "dilation_w": 1, "groups": 1}))
        g.set_inputs([x])
        g.set_outputs([conv])

        # Python interpreter
        py_res = g.evaluate({0: t_x.numpy()})[3]
        np.testing.assert_allclose(py_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

        # C++ interpreter
        cpp_graph = cpp.json_to_graph(g.to_json())
        interp = cpp.Interpreter()
        cpp_res = interp.evaluate(cpp_graph, {0: t_x.numpy()})[3]
        np.testing.assert_allclose(cpp_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

def test_conv2d_3x3_stride2_pad1_no_bias():
    for B in [1, 4]:
        C_in, C_out = 4, 8
        H_in, W_in = 16, 16
        Kh, Kw = 3, 3
        # Output spatial: (16 + 2 - 3)/2 + 1 = 8

        torch.manual_seed(20 + B)
        t_x = torch.randn(B, C_in, H_in, W_in, dtype=torch.float32)
        t_w = torch.randn(C_out, C_in, Kh, Kw, dtype=torch.float32) * 0.1
        t_out = F.conv2d(t_x, t_w, bias=None, stride=2, padding=1)

        g = Graph()
        x = g.add_node(Node(0, "INPUT", shape=[-1, C_in, H_in, W_in]))
        w = g.add_node(Node(1, "CONST", shape=[C_out, C_in, Kh, Kw], const_data=t_w.flatten().tolist()))
        conv = g.add_node(Node(2, "CONV2D", inputs=[x, w], shape=[-1, C_out, 8, 8],
                               conv_attrs={"stride_h": 2, "stride_w": 2, "pad_h": 1, "pad_w": 1,
                                           "dilation_h": 1, "dilation_w": 1, "groups": 1}))
        g.set_inputs([x])
        g.set_outputs([conv])

        py_res = g.evaluate({0: t_x.numpy()})[2]
        np.testing.assert_allclose(py_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

        cpp_graph = cpp.json_to_graph(g.to_json())
        interp = cpp.Interpreter()
        cpp_res = interp.evaluate(cpp_graph, {0: t_x.numpy()})[2]
        np.testing.assert_allclose(cpp_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

def test_conv2d_pointwise_1x1():
    B = 2
    C_in, C_out = 8, 16
    H, W = 7, 7
    torch.manual_seed(33)
    t_x = torch.randn(B, C_in, H, W, dtype=torch.float32)
    t_w = torch.randn(C_out, C_in, 1, 1, dtype=torch.float32) * 0.2
    t_b = torch.randn(C_out, dtype=torch.float32) * 0.1
    t_out = F.conv2d(t_x, t_w, bias=t_b, stride=1, padding=0)

    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, C_in, H, W]))
    w = g.add_node(Node(1, "CONST", shape=[C_out, C_in, 1, 1], const_data=t_w.flatten().tolist()))
    b = g.add_node(Node(2, "CONST", shape=[C_out], const_data=t_b.tolist()))
    conv = g.add_node(Node(3, "CONV2D", inputs=[x, w, b], shape=[-1, C_out, H, W],
                           conv_attrs={"stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0,
                                       "dilation_h": 1, "dilation_w": 1, "groups": 1}))
    g.set_inputs([x])
    g.set_outputs([conv])

    cpp_graph = cpp.json_to_graph(g.to_json())
    interp = cpp.Interpreter()
    cpp_res = interp.evaluate(cpp_graph, {0: t_x.numpy()})[3]
    np.testing.assert_allclose(cpp_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

def test_fused_conv2d_relu():
    B = 3
    C_in, C_out = 3, 6
    H, W = 10, 10
    torch.manual_seed(44)
    t_x = torch.randn(B, C_in, H, W, dtype=torch.float32)
    t_w = torch.randn(C_out, C_in, 3, 3, dtype=torch.float32) * 0.3
    t_b = torch.randn(C_out, dtype=torch.float32) * 0.1

    t_conv = F.conv2d(t_x, t_w, bias=t_b, stride=1, padding=1)
    t_out = F.relu(t_conv)

    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, C_in, H, W]))
    w = g.add_node(Node(1, "CONST", shape=[C_out, C_in, 3, 3], const_data=t_w.flatten().tolist()))
    b = g.add_node(Node(2, "CONST", shape=[C_out], const_data=t_b.tolist()))
    fused = g.add_node(Node(3, "FUSED_CONV2D_RELU", inputs=[x, w, b], shape=[-1, C_out, H, W],
                            conv_attrs={"stride_h": 1, "stride_w": 1, "pad_h": 1, "pad_w": 1,
                                        "dilation_h": 1, "dilation_w": 1, "groups": 1}))
    g.set_inputs([x])
    g.set_outputs([fused])

    py_res = g.evaluate({0: t_x.numpy()})[3]
    np.testing.assert_allclose(py_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

    cpp_graph = cpp.json_to_graph(g.to_json())
    interp = cpp.Interpreter()
    cpp_res = interp.evaluate(cpp_graph, {0: t_x.numpy()})[3]
    np.testing.assert_allclose(cpp_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)
