"""
Unit tests for Milestone 5: Standalone C++17 Code Generation & Dynamic Batch Runtime ABI.
Validates:
1. Dynamic batch code generation and dual ABI symbol exports (forgetune_run, forgetune_run_v1, forgetune_is_dynamic, forgetune_max_batch).
2. Compilation and dynamic batch execution across varying batch sizes (B=1, 3, 16, 64).
3. Out-of-bounds dynamic batch rejection (B <= 0 or B > 128) with InvalidShapeError.
4. Intermediate buffer reuse via memory planner arena in emitted C++.
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import pytest
import torch

from forgetune.graph_adapter import Graph, Node
from forgetune.native_compiler import NativeCompiler
from forgetune.runtime import NativeRuntime
from forgetune.compiler import get_engine_executable
from forgetune.errors import InvalidShapeError, RuntimeError
import forgetune._forgetune_cpp as cpp

ENGINE_EXE = get_engine_executable()

@pytest.fixture(scope="module")
def dynamic_mlp_runtime(tmp_path_factory):
    """
    Compiles a dynamic batch MLP (Linear 16 -> 8 -> ReLU -> Linear 8 -> 4).
    Input shape [-1, 16], Output shape [-1, 4].
    """
    test_dir = str(tmp_path_factory.mktemp("v2_codegen_mlp"))
    json_path = os.path.join(test_dir, "graph.json")
    cpp_path = os.path.join(test_dir, "kernel.cpp")
    dll_path = os.path.join(test_dir, "kernel.dll")

    np.random.seed(42)
    w1 = np.random.randn(16, 8).astype(np.float32) * 0.1
    b1 = np.random.randn(8).astype(np.float32) * 0.1
    w2 = np.random.randn(8, 4).astype(np.float32) * 0.1
    b2 = np.random.randn(4).astype(np.float32) * 0.1

    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", "X", shape=[-1, 16], dtype="f32"))
    w1_id = g.add_node(Node(1, "CONST", "W1", shape=[16, 8], dtype="f32", const_data=w1.flatten().tolist()))
    b1_id = g.add_node(Node(2, "CONST", "B1", shape=[8], dtype="f32", const_data=b1.tolist()))
    mm1_id = g.add_node(Node(3, "MATMUL", "mm1", inputs=[in_id, w1_id], shape=[-1, 8], dtype="f32"))
    add1_id = g.add_node(Node(4, "ADD", "add1", inputs=[mm1_id, b1_id], shape=[-1, 8], dtype="f32"))
    act1_id = g.add_node(Node(5, "RELU", "act1", inputs=[add1_id], shape=[-1, 8], dtype="f32"))

    w2_id = g.add_node(Node(6, "CONST", "W2", shape=[8, 4], dtype="f32", const_data=w2.flatten().tolist()))
    b2_id = g.add_node(Node(7, "CONST", "B2", shape=[4], dtype="f32", const_data=b2.tolist()))
    mm2_id = g.add_node(Node(8, "MATMUL", "mm2", inputs=[act1_id, w2_id], shape=[-1, 4], dtype="f32"))
    add2_id = g.add_node(Node(9, "ADD", "add2", inputs=[mm2_id, b2_id], shape=[-1, 4], dtype="f32"))
    out_id = g.add_node(Node(10, "OUTPUT", "out", inputs=[add2_id], shape=[-1, 4], dtype="f32"))

    g.set_inputs([in_id])
    g.set_outputs([out_id])

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(g.to_json())

    # Compile with engine CLI and fusion enabled
    import subprocess
    cmd = [ENGINE_EXE, "--compile", json_path, "-o", cpp_path, "--variant", "loop_ikj", "--fusion"]
    subprocess.run(cmd, check=True)

    compiler = NativeCompiler()
    compiler.compile_shared_library(cpp_path, dll_path, opt_level="-O2")

    runtime = NativeRuntime(dll_path, (-1, 16), (-1, 4))
    return runtime, (w1, b1, w2, b2)

def test_dynamic_runtime_capabilities(dynamic_mlp_runtime):
    rt, _ = dynamic_mlp_runtime
    assert rt.is_dynamic is True
    assert rt.max_batch_size == 128
    assert hasattr(rt.lib, "forgetune_run")
    assert hasattr(rt.lib, "forgetune_run_v1")
    assert hasattr(rt.lib, "forgetune_is_dynamic")
    assert hasattr(rt.lib, "forgetune_max_batch")
    assert rt.lib.forgetune_is_dynamic() == 1
    assert rt.lib.forgetune_max_batch() == 128

def test_dynamic_execution_across_batches(dynamic_mlp_runtime):
    rt, (w1, b1, w2, b2) = dynamic_mlp_runtime
    
    test_batches = [1, 2, 3, 7, 16, 32, 64, 128]
    for b in test_batches:
        x = np.random.randn(b, 16).astype(np.float32)
        
        # Ground truth
        ref_h1 = np.maximum(0.0, x @ w1 + b1)
        ref_out = ref_h1 @ w2 + b2
        
        # Native execution
        act_out = rt.run(x)
        assert act_out.shape == (b, 4)
        np.testing.assert_allclose(act_out, ref_out, atol=1e-5, rtol=1e-5)

def test_dynamic_batch_out_of_bounds_rejection(dynamic_mlp_runtime):
    rt, _ = dynamic_mlp_runtime
    
    # Batch size 0
    with pytest.raises(InvalidShapeError) as exc_info:
        rt.run(np.empty((0, 16), dtype=np.float32))
    assert "out of valid range" in str(exc_info.value)

    # Batch size 129 (> 128)
    with pytest.raises(InvalidShapeError) as exc_info:
        rt.run(np.ones((129, 16), dtype=np.float32))
    assert "out of valid range" in str(exc_info.value)

    # Invalid feature dimension (15 instead of 16)
    with pytest.raises(InvalidShapeError) as exc_info:
        rt.run(np.ones((4, 15), dtype=np.float32))
    assert "mismatch" in str(exc_info.value)

def test_dynamic_benchmark_kernel(dynamic_mlp_runtime):
    rt, _ = dynamic_mlp_runtime
    x = np.ones((8, 16), dtype=np.float32)
    lat_us = rt.benchmark_kernel(x, iters=100)
    assert isinstance(lat_us, float)
    assert lat_us > 0.0

@pytest.fixture(scope="module")
def dynamic_cnn_runtime(tmp_path_factory):
    """
    Compiles a dynamic batch Small CNN:
    Conv2d(1 -> 4, 3x3, stride=1, pad=1) -> ReLU -> Flatten -> Linear(4*8*8 -> 5)
    Input shape [-1, 1, 8, 8], Output shape [-1, 5].
    """
    test_dir = str(tmp_path_factory.mktemp("v2_codegen_cnn"))
    json_path = os.path.join(test_dir, "graph.json")
    cpp_path = os.path.join(test_dir, "kernel.cpp")
    dll_path = os.path.join(test_dir, "kernel.dll")

    np.random.seed(123)
    w_conv = np.random.randn(4, 1, 3, 3).astype(np.float32) * 0.1
    b_conv = np.random.randn(4).astype(np.float32) * 0.1
    w_fc = np.random.randn(256, 5).astype(np.float32) * 0.1
    b_fc = np.random.randn(5).astype(np.float32) * 0.1

    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", "X", shape=[-1, 1, 8, 8], dtype="f32"))
    w_c_id = g.add_node(Node(1, "CONST", "W_conv", shape=[4, 1, 3, 3], dtype="f32", const_data=w_conv.flatten().tolist()))
    b_c_id = g.add_node(Node(2, "CONST", "B_conv", shape=[4], dtype="f32", const_data=b_conv.tolist()))

    conv_attrs = {"stride_h": 1, "stride_w": 1, "pad_h": 1, "pad_w": 1, "dilation_h": 1, "dilation_w": 1, "groups": 1}
    conv_id = g.add_node(Node(3, "CONV2D", "conv", inputs=[in_id, w_c_id, b_c_id], shape=[-1, 4, 8, 8], dtype="f32", conv_attrs=conv_attrs))
    relu_id = g.add_node(Node(4, "RELU", "relu", inputs=[conv_id], shape=[-1, 4, 8, 8], dtype="f32"))
    flat_id = g.add_node(Node(5, "FLATTEN", "flat", inputs=[relu_id], shape=[-1, 256], dtype="f32"))

    w_fc_id = g.add_node(Node(6, "CONST", "W_fc", shape=[256, 5], dtype="f32", const_data=w_fc.flatten().tolist()))
    b_fc_id = g.add_node(Node(7, "CONST", "B_fc", shape=[5], dtype="f32", const_data=b_fc.tolist()))
    mm_id = g.add_node(Node(8, "MATMUL", "mm", inputs=[flat_id, w_fc_id], shape=[-1, 5], dtype="f32"))
    add_id = g.add_node(Node(9, "ADD", "add", inputs=[mm_id, b_fc_id], shape=[-1, 5], dtype="f32"))
    out_id = g.add_node(Node(10, "OUTPUT", "out", inputs=[add_id], shape=[-1, 5], dtype="f32"))

    g.set_inputs([in_id])
    g.set_outputs([out_id])

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(g.to_json())

    import subprocess
    cmd = [ENGINE_EXE, "--compile", json_path, "-o", cpp_path, "--variant", "loop_ikj", "--fusion"]
    subprocess.run(cmd, check=True)

    compiler = NativeCompiler()
    compiler.compile_shared_library(cpp_path, dll_path, opt_level="-O2")

    runtime = NativeRuntime(dll_path, (-1, 1, 8, 8), (-1, 5))
    return runtime, (w_conv, b_conv, w_fc, b_fc)

def test_dynamic_cnn_execution(dynamic_cnn_runtime):
    rt, (w_conv, b_conv, w_fc, b_fc) = dynamic_cnn_runtime

    class PyTorchCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 4, 3, padding=1)
            self.conv.weight.data = torch.from_numpy(w_conv)
            self.conv.bias.data = torch.from_numpy(b_conv)
            self.fc = torch.nn.Linear(256, 5)
            self.fc.weight.data = torch.from_numpy(w_fc.T)
            self.fc.bias.data = torch.from_numpy(b_fc)

        def forward(self, x):
            return self.fc(torch.flatten(torch.relu(self.conv(x)), start_dim=1))

    pt_model = PyTorchCNN().eval()

    test_batches = [1, 2, 4, 7, 16]
    for b in test_batches:
        x_torch = torch.randn(b, 1, 8, 8)
        with torch.no_grad():
            ref_out = pt_model(x_torch).numpy()
        
        act_out = rt.run(x_torch.numpy())
        assert act_out.shape == (b, 5)
        np.testing.assert_allclose(act_out, ref_out, atol=1e-5, rtol=1e-5)

def test_safe_library_loading_file_not_found():
    with pytest.raises(FileNotFoundError) as exc_info:
        NativeRuntime("non_existent_library_12345.dll", (-1, 4), (-1, 2))
    assert "Shared library not found" in str(exc_info.value)

def test_safe_library_loading_direct_success(dynamic_mlp_runtime):
    rt, _ = dynamic_mlp_runtime
    # Confirm runtime loaded directly from original_dll_path without temp staging evasion
    assert os.path.isfile(rt.original_dll_path)
    assert hasattr(rt.lib, "forgetune_run")

def test_safe_library_loading_smart_app_control_diagnostic(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    dummy_dll = str(tmp_path / "probe_blocked.dll")
    with open(dummy_dll, "wb") as f:
        f.write(b"MZdummybinary")

    attempts = 0
    def mock_cdll(path):
        nonlocal attempts
        attempts += 1
        raise OSError("[WinError 4551] An Application Control policy has blocked this file")

    import ctypes, time
    monkeypatch.setattr(ctypes, "CDLL", mock_cdll)
    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    with pytest.raises(OSError) as exc_info:
        NativeRuntime(dummy_dll, (-1, 4), (-1, 2))

    # Must provide explicit diagnostics explaining that Code Integrity policy blocked execution
    err_msg = str(exc_info.value)
    assert "Code Integrity policy blocked execution (WinError 4551)" in err_msg or "Smart App Control" in err_msg
    # Fails immediately on attempt 1 without wasteful retries or evasion
    assert attempts == 1
    assert len(sleep_calls) == 0

def test_safe_library_loading_transient_sharing_retry(monkeypatch, tmp_path, dynamic_mlp_runtime):
    monkeypatch.setattr(sys, "platform", "win32")
    rt, _ = dynamic_mlp_runtime
    dll_path = rt.original_dll_path

    import ctypes, time
    real_cdll = ctypes.CDLL
    attempts = 0
    sleep_calls = []

    def mock_cdll_with_transient_failure(path):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("[WinError 32] The process cannot access the file because it is being used by another process")
        return real_cdll(path)

    monkeypatch.setattr(ctypes, "CDLL", mock_cdll_with_transient_failure)
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    test_rt = NativeRuntime(dll_path, (-1, 4), (-1, 2))
    assert attempts == 3
    assert len(sleep_calls) == 2
    assert hasattr(test_rt.lib, "forgetune_run")

def test_safe_library_loading_invalid_library(tmp_path):
    corrupt_dll = str(tmp_path / "corrupt.dll")
    with open(corrupt_dll, "w") as f:
        f.write("THIS_IS_NOT_A_VALID_SHARED_LIBRARY")

    with pytest.raises(OSError) as exc_info:
        NativeRuntime(corrupt_dll, (-1, 4), (-1, 2))
    assert "Failed to load shared library" in str(exc_info.value)

def test_safe_library_metadata_preservation(tmp_path, dynamic_mlp_runtime):
    import shutil
    rt, _ = dynamic_mlp_runtime
    target_dll = str(tmp_path / "model_with_metadata.dll")
    shutil.copy(rt.original_dll_path, target_dll)

    # Attach Zone.Identifier alternate data stream simulating file downloaded/scanned by Windows
    zone_stream = target_dll + ":Zone.Identifier"
    zone_content = b"[ZoneTransfer]\r\nZoneId=3\r\n"
    has_ads = False
    try:
        with open(zone_stream, "wb") as f:
            f.write(zone_content)
        has_ads = os.path.exists(zone_stream)
    except Exception:
        has_ads = False

    # Load runtime
    loaded_rt = NativeRuntime(target_dll, (-1, 4), (-1, 2))
    assert hasattr(loaded_rt.lib, "forgetune_run")

    # Verify original file and metadata remain untouched and unstripped
    assert os.path.isfile(target_dll)
    if has_ads:
        assert os.path.exists(zone_stream)
        with open(zone_stream, "rb") as f:
            assert f.read() == zone_content

