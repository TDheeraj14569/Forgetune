"""
Unit tests for the pybind11 C++ bridge (_forgetune_cpp).
Validates that C++ IR, Verifier, and Interpreter are accessible directly in Python
and produce identical numerical results to PyTorch.
"""

import pytest
import torch
import numpy as np
from forgetune import _forgetune_cpp

def test_pybind11_ir_and_verifier():
    g = _forgetune_cpp.IRGraph()
    assert g.live_node_count() == 0

    x_meta = _forgetune_cpp.TensorMeta(_forgetune_cpp.TensorShape([2, 4]), _forgetune_cpp.DType.FLOAT32)
    x = g.add_node(_forgetune_cpp.IRNode(0, _forgetune_cpp.OpKind.INPUT, "x", [], x_meta))

    w_meta = _forgetune_cpp.TensorMeta(_forgetune_cpp.TensorShape([4, 3]), _forgetune_cpp.DType.FLOAT32)
    w_node = _forgetune_cpp.IRNode(1, _forgetune_cpp.OpKind.CONST, "w", [], w_meta)
    w_node.const_data = [1.0] * 12
    w = g.add_node(w_node)

    out_meta = _forgetune_cpp.TensorMeta(_forgetune_cpp.TensorShape([2, 3]), _forgetune_cpp.DType.FLOAT32)
    mm = g.add_node(_forgetune_cpp.IRNode(2, _forgetune_cpp.OpKind.MATMUL, "mm", [x, w], out_meta))

    g.set_inputs([x])
    g.set_outputs([mm])

    vres = _forgetune_cpp.verify_graph(g, check_throw=False)
    assert vres.passed, f"Graph verification failed: {vres.to_string()}"
    assert g.live_node_count() == 3

def test_pybind11_interpreter_against_pytorch():
    # Build: Y = ReLU(X @ W + B)
    # Compare C++ Interpreter directly with PyTorch
    torch.manual_seed(42)
    x_torch = torch.randn(2, 4)
    w_torch = torch.randn(4, 3)
    b_torch = torch.randn(3)

    # PyTorch computation
    expected_out = torch.relu(x_torch @ w_torch + b_torch).numpy()

    # C++ IR via pybind11
    g = _forgetune_cpp.IRGraph()
    x_id = g.add_node(_forgetune_cpp.IRNode(0, _forgetune_cpp.OpKind.INPUT, "x", [],
        _forgetune_cpp.TensorMeta(_forgetune_cpp.TensorShape([2, 4]), _forgetune_cpp.DType.FLOAT32)))

    w_node = _forgetune_cpp.IRNode(1, _forgetune_cpp.OpKind.CONST, "w", [],
        _forgetune_cpp.TensorMeta(_forgetune_cpp.TensorShape([4, 3]), _forgetune_cpp.DType.FLOAT32))
    w_node.const_data = w_torch.flatten().tolist()
    w_id = g.add_node(w_node)

    b_node = _forgetune_cpp.IRNode(2, _forgetune_cpp.OpKind.CONST, "b", [],
        _forgetune_cpp.TensorMeta(_forgetune_cpp.TensorShape([3]), _forgetune_cpp.DType.FLOAT32))
    b_node.const_data = b_torch.flatten().tolist()
    b_id = g.add_node(b_node)

    mm_id = g.add_node(_forgetune_cpp.IRNode(3, _forgetune_cpp.OpKind.MATMUL, "mm", [x_id, w_id],
        _forgetune_cpp.TensorMeta(_forgetune_cpp.TensorShape([2, 3]), _forgetune_cpp.DType.FLOAT32)))

    add_id = g.add_node(_forgetune_cpp.IRNode(4, _forgetune_cpp.OpKind.ADD, "add", [mm_id, b_id],
        _forgetune_cpp.TensorMeta(_forgetune_cpp.TensorShape([2, 3]), _forgetune_cpp.DType.FLOAT32)))

    relu_id = g.add_node(_forgetune_cpp.IRNode(5, _forgetune_cpp.OpKind.RELU, "relu", [add_id],
        _forgetune_cpp.TensorMeta(_forgetune_cpp.TensorShape([2, 3]), _forgetune_cpp.DType.FLOAT32)))

    g.set_inputs([x_id])
    g.set_outputs([relu_id])

    # Run C++ reference interpreter exposed to Python
    interp = _forgetune_cpp.Interpreter()
    out_dict = interp.evaluate(g, {x_id: x_torch.numpy().astype(np.float32)})

    cpp_out = out_dict[relu_id]
    assert cpp_out.shape == (2, 3)
    max_diff = float(np.max(np.abs(cpp_out - expected_out)))
    assert max_diff < 1e-5, f"C++ interpreter mismatch vs PyTorch: {max_diff}"

def test_pybind11_json_roundtrip():
    g = _forgetune_cpp.IRGraph()
    x = g.add_node(_forgetune_cpp.IRNode(0, _forgetune_cpp.OpKind.INPUT, "x", [],
        _forgetune_cpp.TensorMeta(_forgetune_cpp.TensorShape([2, 2]), _forgetune_cpp.DType.FLOAT32)))
    g.set_inputs([x])
    g.set_outputs([x])

    json_str = _forgetune_cpp.graph_to_json(g)
    assert "schema_version" in json_str

    g2 = _forgetune_cpp.json_to_graph(json_str)
    assert g2.live_node_count() == 1
    assert g2.get_inputs() == [x]
    assert g2.get_outputs() == [x]

def test_pybind11_native_cpp_test_suite():
    """
    Executes the 13 native C++ unit tests in-process via pybind11.
    Verifies that all native C++ tests pass with returncode 0.
    """
    res = _forgetune_cpp.run_cpp_test_suite()
    assert res == 0, f"Native C++ test suite failed with exit code {res}"

def test_pybind11_compiler_and_codegen():
    """
    Validates Compiler and CodeGenerator pybind11 bindings.
    Compiles an IRGraph to C++ source directly in-process without CLI.
    """
    g = _forgetune_cpp.IRGraph()
    x = g.add_node(_forgetune_cpp.IRNode(0, _forgetune_cpp.OpKind.INPUT, "x", [],
        _forgetune_cpp.TensorMeta(_forgetune_cpp.TensorShape([2, 4]), _forgetune_cpp.DType.FLOAT32)))
    w_node = _forgetune_cpp.IRNode(1, _forgetune_cpp.OpKind.CONST, "w", [],
        _forgetune_cpp.TensorMeta(_forgetune_cpp.TensorShape([4, 4]), _forgetune_cpp.DType.FLOAT32))
    w_node.const_data = [1.0] * 16
    w = g.add_node(w_node)
    out = g.add_node(_forgetune_cpp.IRNode(2, _forgetune_cpp.OpKind.MATMUL, "mm", [x, w],
        _forgetune_cpp.TensorMeta(_forgetune_cpp.TensorShape([2, 4]), _forgetune_cpp.DType.FLOAT32)))
    g.set_inputs([x])
    g.set_outputs([out])

    config = _forgetune_cpp.KernelConfig()
    config.candidate_id = "test_candidate"
    config.matmul_variant = _forgetune_cpp.MatMulVariant.LOOP_IKJ

    compiler = _forgetune_cpp.Compiler(config)
    res = compiler.compile(g)
    assert res.success, f"Compiler failed: {res.error_message}"
    assert "forgetune_run" in res.generated_cpp_source
    assert "LOOP_IKJ" in res.generated_cpp_source or "loop_ikj" in res.generated_cpp_source.lower()

