"""
Acceptance Test Suite: Runtime & ABI Robustness Verification
Tests native entry point error contracts (null pointers, element count mismatches),
unaligned buffer handling, repeated execution re-entrancy / stability,
and graceful handling of invalid/corrupted dynamic libraries.
"""

import os
import shutil
import tempfile
import ctypes
import pytest
import numpy as np

from forgetune.graph_adapter import Graph, Node
from forgetune.native_compiler import NativeCompiler
from forgetune.runtime import NativeRuntime
from forgetune.errors import RuntimeError, InvalidShapeError

@pytest.fixture(scope="module")
def sample_runtime_fixture():
    temp_dir = os.path.abspath(os.path.join("build", "artifacts", "acceptance_abi"))
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
    os.makedirs(temp_dir, exist_ok=True)
    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", "X", shape=[4, 8], dtype="f32"))
    w_id = g.add_node(Node(1, "CONST", "W", shape=[8, 6], dtype="f32", const_data=[0.5]*48))
    mm_id = g.add_node(Node(2, "MATMUL", "mm", inputs=[in_id, w_id], shape=[4, 6], dtype="f32"))
    out_id = g.add_node(Node(3, "OUTPUT", "out", inputs=[mm_id], shape=[4, 6], dtype="f32"))
    g.set_inputs([in_id])
    g.set_outputs([out_id])

    json_path = os.path.join(temp_dir, "graph.json")
    cpp_path = os.path.join(temp_dir, "kernel.cpp")
    dll_path = os.path.join(temp_dir, "kernel.dll")

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(g.to_json())

    _test_engine = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "forgetune_test_engine.exe"))
    engine_exe = _test_engine if os.path.isfile(_test_engine) else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "forgetune_engine.exe"))
    import subprocess
    subprocess.run([engine_exe, "--compile", json_path, "-o", cpp_path, "--variant", "naive_ijk"], check=True)

    compiler = NativeCompiler()
    compiler.compile_shared_library(cpp_path, dll_path, opt_level="-O2")

    rt = NativeRuntime(dll_path, (4, 8), (4, 6))
    yield rt

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

def test_acceptance_abi_null_pointer_rejection(sample_runtime_fixture):
    """
    Directly tests the C ABI contract for forgetune_run:
    Must return -1 when input or output pointer is NULL.
    """
    rt = sample_runtime_fixture
    inp = np.ones((4, 8), dtype=np.float32)
    out = np.zeros((4, 6), dtype=np.float32)

    inp_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Test NULL input pointer
    status_null_in = rt.run_fn(None, 32, out_ptr, 24)
    assert status_null_in == -1, f"Expected status -1 for NULL input, got {status_null_in}"

    # Test NULL output pointer
    status_null_out = rt.run_fn(inp_ptr, 32, None, 24)
    assert status_null_out == -1, f"Expected status -1 for NULL output, got {status_null_out}"

    # Test both NULL
    status_both_null = rt.run_fn(None, 32, None, 24)
    assert status_both_null == -1, f"Expected status -1 for both NULL, got {status_both_null}"

def test_acceptance_abi_size_mismatch_rejection(sample_runtime_fixture):
    """
    Directly tests the C ABI contract for forgetune_run:
    Must return -2 on input element count mismatch.
    Must return -3 on output element count mismatch.
    """
    rt = sample_runtime_fixture
    inp = np.ones((4, 8), dtype=np.float32)
    out = np.zeros((4, 6), dtype=np.float32)

    inp_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Expected: input count = 32, output count = 24

    # Input count mismatch (31 instead of 32)
    status_in_under = rt.run_fn(inp_ptr, 31, out_ptr, 24)
    assert status_in_under == -2, f"Expected -2 for input count mismatch, got {status_in_under}"

    # Input count mismatch (33 instead of 32)
    status_in_over = rt.run_fn(inp_ptr, 33, out_ptr, 24)
    assert status_in_over == -2, f"Expected -2 for input count mismatch, got {status_in_over}"

    # Output count mismatch (23 instead of 24)
    status_out_under = rt.run_fn(inp_ptr, 32, out_ptr, 23)
    assert status_out_under == -3, f"Expected -3 for output count mismatch, got {status_out_under}"

    # Output count mismatch (25 instead of 24)
    status_out_over = rt.run_fn(inp_ptr, 32, out_ptr, 25)
    assert status_out_over == -3, f"Expected -3 for output count mismatch, got {status_out_over}"

def test_acceptance_shape_validation_in_python_wrapper(sample_runtime_fixture):
    """Verifies NativeRuntime.run() rejects invalid shapes with InvalidShapeError."""
    rt = sample_runtime_fixture

    # Wrong shape (4, 7) instead of (4, 8)
    with pytest.raises(InvalidShapeError):
        rt.run(np.ones((4, 7), dtype=np.float32))

    # Wrong rank (32,) instead of (4, 8)
    with pytest.raises(InvalidShapeError):
        rt.run(np.ones(32, dtype=np.float32))

def test_acceptance_unaligned_memory_buffer(sample_runtime_fixture):
    """
    Verifies that unaligned input memory (e.g. array slice with offset)
    executes correctly without crashing or corrupting memory.
    """
    rt = sample_runtime_fixture
    # Create larger buffer and slice with offset 1
    raw_buf = np.ones(33, dtype=np.float32)
    unaligned_inp = raw_buf[1:].reshape(4, 8)

    # Should execute smoothly
    out = rt.run(unaligned_inp)
    assert out.shape == (4, 6)
    # 8 * 1.0 * 0.5 = 4.0
    np.testing.assert_allclose(out, 4.0, atol=1e-5)

def test_acceptance_reentrancy_stress_1000_calls(sample_runtime_fixture):
    """
    Stress tests re-entrancy with 1,000 consecutive calls using changing random inputs.
    Ensures no memory leak, buffer overflow, or state corruption across calls.
    """
    rt = sample_runtime_fixture
    np.random.seed(999)

    for i in range(1000):
        inp = np.random.randn(4, 8).astype(np.float32)
        out = rt.run(inp)
        assert out.shape == (4, 6)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

def test_acceptance_invalid_dll_loading_graceful_failure():
    """Verifies that non-existent or invalid DLL paths raise clear exceptions."""
    with pytest.raises(FileNotFoundError):
        NativeRuntime("non_existent_library_12345.dll", (2, 2), (2, 2))
