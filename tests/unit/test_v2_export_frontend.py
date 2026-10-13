"""
Unit tests for ForgeTune v2.0 PyTorch Frontend integration.
Tests ahead-of-time export with dynamic batching on:
- Model A: Dynamic TinyMLP
- Model B: Residual MLP
- Model C: Small CNN
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from forgetune.export_frontend import export_to_forgetune_ir
from forgetune import _forgetune_cpp as cpp

# Model A: TinyMLP
class TinyMLP(nn.Module):
    def __init__(self, in_features=16, hidden=32, out_features=8):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Model B: Residual MLP
class ResidualMLP(nn.Module):
    def __init__(self, features=16):
        super().__init__()
        self.fc1 = nn.Linear(features, features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(features, features)
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.fc2(h)
        res = h + x
        return self.tanh(res)

# Model C: Small CNN
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 16 * 16, 10)

    def forward(self, x):
        return self.fc(self.flatten(self.relu(self.conv(x))))

def test_export_tinymlp_dynamic_batch():
    model = TinyMLP()
    example_input = torch.randn(2, 16, dtype=torch.float32)

    ir_graph = export_to_forgetune_ir(model, (example_input,), dynamic_batch=True)
    json_str = ir_graph.to_json()
    cpp_graph = cpp.json_to_graph(json_str)

    res = cpp.verify_graph(cpp_graph, False)
    assert res.passed, f"Verification failed: {res.to_string()}"

    interp = cpp.Interpreter()

    # Test differential correctness across multiple batch sizes
    for B in [1, 3, 7, 16]:
        test_x = torch.randn(B, 16, dtype=torch.float32)
        expected = model(test_x).detach().numpy()

        py_out = list(ir_graph.evaluate({ir_graph.inputs[0]: test_x.numpy()}).values())[0]
        np.testing.assert_allclose(py_out, expected, atol=1e-5, rtol=1e-4)

        cpp_out = list(interp.evaluate(cpp_graph, {ir_graph.inputs[0]: test_x.numpy()}).values())[0]
        np.testing.assert_allclose(cpp_out, expected, atol=1e-5, rtol=1e-4)

def test_export_residual_mlp_dynamic_batch():
    model = ResidualMLP()
    example_input = torch.randn(4, 16, dtype=torch.float32)

    ir_graph = export_to_forgetune_ir(model, (example_input,), dynamic_batch=True)
    json_str = ir_graph.to_json()
    cpp_graph = cpp.json_to_graph(json_str)

    res = cpp.verify_graph(cpp_graph, False)
    assert res.passed, f"Verification failed: {res.to_string()}"

    interp = cpp.Interpreter()

    for B in [1, 2, 5, 12]:
        test_x = torch.randn(B, 16, dtype=torch.float32)
        expected = model(test_x).detach().numpy()

        py_out = list(ir_graph.evaluate({ir_graph.inputs[0]: test_x.numpy()}).values())[0]
        np.testing.assert_allclose(py_out, expected, atol=1e-5, rtol=1e-4)

        cpp_out = list(interp.evaluate(cpp_graph, {ir_graph.inputs[0]: test_x.numpy()}).values())[0]
        np.testing.assert_allclose(cpp_out, expected, atol=1e-5, rtol=1e-4)

def test_export_small_cnn_dynamic_batch():
    model = SmallCNN()
    example_input = torch.randn(2, 3, 16, 16, dtype=torch.float32)

    ir_graph = export_to_forgetune_ir(model, (example_input,), dynamic_batch=True)
    json_str = ir_graph.to_json()
    cpp_graph = cpp.json_to_graph(json_str)

    res = cpp.verify_graph(cpp_graph, False)
    assert res.passed, f"Verification failed: {res.to_string()}"

    interp = cpp.Interpreter()

    for B in [1, 2, 4]:
        test_x = torch.randn(B, 3, 16, 16, dtype=torch.float32)
        expected = model(test_x).detach().numpy()

        py_out = list(ir_graph.evaluate({ir_graph.inputs[0]: test_x.numpy()}).values())[0]
        np.testing.assert_allclose(py_out, expected, atol=1e-4, rtol=1e-4)

        cpp_out = list(interp.evaluate(cpp_graph, {ir_graph.inputs[0]: test_x.numpy()}).values())[0]
        np.testing.assert_allclose(cpp_out, expected, atol=1e-4, rtol=1e-4)

def test_export_preserves_static_batch_default():
    model = TinyMLP()
    example_input = torch.randn(4, 16, dtype=torch.float32)

    # dynamic_batch defaults to False for backward compatibility
    ir_graph = export_to_forgetune_ir(model, (example_input,))
    assert ir_graph.nodes[ir_graph.inputs[0]].shape == [4, 16]
