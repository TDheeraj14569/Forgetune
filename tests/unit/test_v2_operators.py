"""
Unit tests for ForgeTune v2.0 Operators: MUL, SIGMOID, TANH, FLATTEN.
Compares both Python and C++ reference interpreters against PyTorch ground truth.
"""

import pytest
import torch
import numpy as np
from forgetune.graph_adapter import Node, Graph
from forgetune import _forgetune_cpp as cpp

TOL_ATOL = 1e-5
TOL_RTOL = 1e-4

def test_v2_mul_exact_elementwise():
    for B in [1, 4, 13]:
        g = Graph()
        x1 = g.add_node(Node(0, "INPUT", shape=[-1, 16]))
        x2 = g.add_node(Node(1, "INPUT", shape=[-1, 16]))
        mul = g.add_node(Node(2, "MUL", inputs=[x1, x2], shape=[-1, 16]))
        g.set_inputs([x1, x2])
        g.set_outputs([mul])

        torch.manual_seed(42 + B)
        t_a = torch.randn(B, 16, dtype=torch.float32)
        t_b = torch.randn(B, 16, dtype=torch.float32)
        t_out = t_a * t_b

        # Python interpreter
        py_res = g.evaluate({0: t_a.numpy(), 1: t_b.numpy()})[2]
        np.testing.assert_allclose(py_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

        # C++ interpreter
        cpp_graph = cpp.json_to_graph(g.to_json())
        interp = cpp.Interpreter()
        cpp_res = interp.evaluate(cpp_graph, {0: t_a.numpy(), 1: t_b.numpy()})[2]
        np.testing.assert_allclose(cpp_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

def test_v2_mul_broadcast_vector():
    for B in [1, 7, 32]:
        g = Graph()
        x = g.add_node(Node(0, "INPUT", shape=[-1, 24]))
        v = g.add_node(Node(1, "CONST", shape=[24], const_data=[0.5] * 24))
        mul = g.add_node(Node(2, "MUL", inputs=[x, v], shape=[-1, 24]))
        g.set_inputs([x])
        g.set_outputs([mul])

        torch.manual_seed(100 + B)
        t_x = torch.randn(B, 24, dtype=torch.float32)
        t_v = torch.full((24,), 0.5, dtype=torch.float32)
        t_out = t_x * t_v

        # Python interpreter
        py_res = g.evaluate({0: t_x.numpy()})[2]
        np.testing.assert_allclose(py_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

        # C++ interpreter
        cpp_graph = cpp.json_to_graph(g.to_json())
        interp = cpp.Interpreter()
        cpp_res = interp.evaluate(cpp_graph, {0: t_x.numpy()})[2]
        np.testing.assert_allclose(cpp_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

def test_v2_mul_broadcast_channel_4d():
    for B in [1, 3]:
        g = Graph()
        x = g.add_node(Node(0, "INPUT", shape=[-1, 4, 8, 8]))
        c_weights = [0.1, 0.2, -0.5, 1.2]
        c = g.add_node(Node(1, "CONST", shape=[1, 4, 1, 1], const_data=c_weights))
        mul = g.add_node(Node(2, "MUL", inputs=[x, c], shape=[-1, 4, 8, 8]))
        g.set_inputs([x])
        g.set_outputs([mul])

        torch.manual_seed(200 + B)
        t_x = torch.randn(B, 4, 8, 8, dtype=torch.float32)
        t_c = torch.tensor(c_weights, dtype=torch.float32).view(1, 4, 1, 1)
        t_out = t_x * t_c

        # Python interpreter
        py_res = g.evaluate({0: t_x.numpy()})[2]
        np.testing.assert_allclose(py_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

        # C++ interpreter
        cpp_graph = cpp.json_to_graph(g.to_json())
        interp = cpp.Interpreter()
        cpp_res = interp.evaluate(cpp_graph, {0: t_x.numpy()})[2]
        np.testing.assert_allclose(cpp_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

def test_v2_sigmoid_semantics():
    for B in [1, 5, 16]:
        g = Graph()
        x = g.add_node(Node(0, "INPUT", shape=[-1, 20]))
        sig = g.add_node(Node(1, "SIGMOID", inputs=[x], shape=[-1, 20]))
        g.set_inputs([x])
        g.set_outputs([sig])

        # Range includes zeros, positive, negative, and extreme saturation values
        t_x = torch.linspace(-10.0, 10.0, B * 20, dtype=torch.float32).view(B, 20)
        t_out = torch.sigmoid(t_x)

        py_res = g.evaluate({0: t_x.numpy()})[1]
        np.testing.assert_allclose(py_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

        cpp_graph = cpp.json_to_graph(g.to_json())
        interp = cpp.Interpreter()
        cpp_res = interp.evaluate(cpp_graph, {0: t_x.numpy()})[1]
        np.testing.assert_allclose(cpp_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

def test_v2_tanh_semantics():
    for B in [1, 4, 16]:
        g = Graph()
        x = g.add_node(Node(0, "INPUT", shape=[-1, 15]))
        tnh = g.add_node(Node(1, "TANH", inputs=[x], shape=[-1, 15]))
        g.set_inputs([x])
        g.set_outputs([tnh])

        t_x = torch.linspace(-6.0, 6.0, B * 15, dtype=torch.float32).view(B, 15)
        t_out = torch.tanh(t_x)

        py_res = g.evaluate({0: t_x.numpy()})[1]
        np.testing.assert_allclose(py_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

        cpp_graph = cpp.json_to_graph(g.to_json())
        interp = cpp.Interpreter()
        cpp_res = interp.evaluate(cpp_graph, {0: t_x.numpy()})[1]
        np.testing.assert_allclose(cpp_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

def test_v2_flatten_semantics():
    for B in [1, 2, 8, 32]:
        g = Graph()
        x = g.add_node(Node(0, "INPUT", shape=[-1, 8, 4, 4]))
        flat = g.add_node(Node(1, "FLATTEN", inputs=[x], shape=[-1, 8 * 4 * 4]))
        g.set_inputs([x])
        g.set_outputs([flat])

        torch.manual_seed(300 + B)
        t_x = torch.randn(B, 8, 4, 4, dtype=torch.float32)
        t_out = torch.flatten(t_x, start_dim=1)

        assert t_out.shape == (B, 128)

        py_res = g.evaluate({0: t_x.numpy()})[1]
        assert py_res.shape == (B, 128)
        np.testing.assert_allclose(py_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

        cpp_graph = cpp.json_to_graph(g.to_json())
        interp = cpp.Interpreter()
        cpp_res = interp.evaluate(cpp_graph, {0: t_x.numpy()})[1]
        assert cpp_res.shape == (B, 128)
        np.testing.assert_allclose(cpp_res, t_out.numpy(), atol=TOL_ATOL, rtol=TOL_RTOL)

def test_v2_dynamic_batch_rejection_out_of_bounds():
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, 16]))
    relu = g.add_node(Node(1, "RELU", inputs=[x], shape=[-1, 16]))
    g.set_inputs([x])
    g.set_outputs([relu])

    cpp_graph = cpp.json_to_graph(g.to_json())
    interp = cpp.Interpreter()

    # Batch 129 exceeds DEFAULT_MAX_BATCH (128)
    data_too_large = np.zeros((129, 16), dtype=np.float32)
    with pytest.raises(Exception):
        interp.evaluate(cpp_graph, {0: data_too_large})

    # Unaligned buffer size (not a multiple of static element count)
    data_unaligned = np.zeros((17,), dtype=np.float32)
    with pytest.raises(Exception):
        interp.evaluate(cpp_graph, {0: data_unaligned})
