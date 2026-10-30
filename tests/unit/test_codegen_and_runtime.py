"""
Category D: Code-Generation and Native Runtime Tests.
Validates code generation, symbol export, length validation, null-pointer checks.
"""

import os
import subprocess
import pytest
import numpy as np
from forgetune.graph_adapter import Graph, Node
from forgetune.native_compiler import NativeCompiler
from forgetune.runtime import NativeRuntime
from forgetune.compiler import get_engine_executable
from forgetune.errors import InvalidShapeError

ENGINE_EXE = get_engine_executable()

@pytest.fixture(scope="module")
def compiled_runtime(tmp_path_factory):
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[2, 4]))
    w = g.add_node(Node(1, "CONST", shape=[4, 3], const_data=[1.0] * 12))
    mm = g.add_node(Node(2, "MATMUL", inputs=[x, w], shape=[2, 3]))
    g.set_inputs([x])
    g.set_outputs([mm])

    test_dir = str(tmp_path_factory.mktemp("runtime_fixture"))
    json_path = os.path.join(test_dir, "graph.json")
    cpp_path = os.path.join(test_dir, "kernel.cpp")
    dll_path = os.path.join(test_dir, "kernel.dll")

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(g.to_json())

    subprocess.run([ENGINE_EXE, "--compile", json_path, "-o", cpp_path], check=True)
    compiler = NativeCompiler()
    compiler.compile_shared_library(cpp_path, dll_path)

    return NativeRuntime(dll_path, (2, 4), (2, 3))

def test_runtime_successful_execution(compiled_runtime):
    inp = np.ones((2, 4), dtype=np.float32) * 2.0
    out = compiled_runtime.run(inp)
    assert out.shape == (2, 3)
    # [2.0] * 4 dot [1.0] * 4 = 8.0
    assert np.allclose(out, 8.0)

def test_runtime_invalid_shape_rejection(compiled_runtime):
    inp_wrong = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(InvalidShapeError):
        compiled_runtime.run(inp_wrong)

def test_runtime_null_pointer_contract(compiled_runtime):
    # Call directly with null pointer to check status code -1
    out = np.zeros((2, 3), dtype=np.float32)
    out_ptr = out.ctypes.data_as(compiled_runtime.run_fn.argtypes[2])
    status = compiled_runtime.run_fn(None, 8, out_ptr, 6)
    assert status == -1, "Expected -1 for null input pointer"

def test_runtime_size_mismatch_contract(compiled_runtime):
    inp = np.ones((2, 4), dtype=np.float32)
    out = np.zeros((2, 3), dtype=np.float32)
    inp_ptr = inp.ctypes.data_as(compiled_runtime.run_fn.argtypes[0])
    out_ptr = out.ctypes.data_as(compiled_runtime.run_fn.argtypes[2])
    # Pass incorrect size (e.g. 7 instead of 8)
    status = compiled_runtime.run_fn(inp_ptr, 7, out_ptr, 6)
    assert status == -2, "Expected -2 for input size mismatch"

def test_native_execution_both_bias_forms(tmp_path):
    compiler = NativeCompiler()
    
    # 1. Rank-1 bias [N]
    g1 = Graph()
    x1 = g1.add_node(Node(0, "INPUT", shape=[2, 3]))
    b1 = g1.add_node(Node(1, "CONST", shape=[3], const_data=[10.0, 20.0, 30.0]))
    add1 = g1.add_node(Node(2, "ADD", inputs=[x1, b1], shape=[2, 3]))
    g1.set_inputs([x1])
    g1.set_outputs([add1])
    
    c1_json = str(tmp_path / "rank1_bias.json")
    c1_cpp = str(tmp_path / "rank1_bias.cpp")
    c1_dll = str(tmp_path / "rank1_bias.dll")
    with open(c1_json, "w") as f:
        f.write(g1.to_json())
    subprocess.run([ENGINE_EXE, "--compile", c1_json, "-o", c1_cpp], check=True)
    compiler.compile_shared_library(c1_cpp, c1_dll)
    rt1 = NativeRuntime(c1_dll, (2, 3), (2, 3))
    
    inp = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    out1 = rt1.run(inp)
    expected1 = inp + np.array([10.0, 20.0, 30.0], dtype=np.float32)
    assert np.allclose(out1, expected1), "Rank-1 [N] bias native execution mismatch"
    
    # 2. Rank-2 bias [1, N]
    g2 = Graph()
    x2 = g2.add_node(Node(0, "INPUT", shape=[2, 3]))
    b2 = g2.add_node(Node(1, "CONST", shape=[1, 3], const_data=[100.0, 200.0, 300.0]))
    add2 = g2.add_node(Node(2, "ADD", inputs=[x2, b2], shape=[2, 3]))
    g2.set_inputs([x2])
    g2.set_outputs([add2])
    
    c2_json = str(tmp_path / "rank2_bias.json")
    c2_cpp = str(tmp_path / "rank2_bias.cpp")
    c2_dll = str(tmp_path / "rank2_bias.dll")
    with open(c2_json, "w") as f:
        f.write(g2.to_json())
    subprocess.run([ENGINE_EXE, "--compile", c2_json, "-o", c2_cpp], check=True)
    compiler.compile_shared_library(c2_cpp, c2_dll)
    rt2 = NativeRuntime(c2_dll, (2, 3), (2, 3))
    
    out2 = rt2.run(inp)
    expected2 = inp + np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
    assert np.allclose(out2, expected2), "Rank-2 [1, N] bias native execution mismatch"

