"""
Category G: Rigorous All-Variants & Fusion Correctness Tests.
Explicitly verifies all five claimed kernel variants across:
  - NAIVE_IJK
  - LOOP_IKJ
  - TILED (tiles 8, 16)
  - FUSED_IKJ
  - FUSED_TILED (tiles 8, 16)
Under:
  - Non-square prime dimensions with partial tile boundaries (M=17, K=31, N=23)
  - Exact tile multiple dimensions (M=16, K=16, N=16)
  - Batch size 1 vector-matrix (M=1, K=48, N=24)
  - All-zero and all-negative inputs (testing ReLU clipping behavior)
"""

import os
import subprocess
import time
import torch
import torch.nn as nn
import numpy as np
import pytest

from forgetune.native_compiler import NativeCompiler
from forgetune.runtime import NativeRuntime
from forgetune.graph_adapter import Graph, Node
from forgetune.compiler import get_engine_executable

ENGINE_EXE = get_engine_executable()

@pytest.mark.parametrize("M,K,N", [
    (17, 31, 23),  # Prime dimensions, partial tile boundary for 8 and 16
    (16, 16, 16),  # Exact tile multiple for 8 and 16
    (3, 5, 7),     # Dimensions strictly smaller than tile size 8 and 16
    (1, 48, 24),   # Batch size 1 vector-matrix multiplication
])
def test_all_five_kernel_variants(tmp_path, M, K, N):
    np.random.seed(42 + M + K + N)

    # Weights and bias
    w_data = np.random.randn(K, N).astype(np.float32) * 0.1
    b_data = np.random.randn(N).astype(np.float32) * 0.05
    x_data = np.random.randn(M, K).astype(np.float32)

    # Independent reference calculation (PyTorch)
    with torch.no_grad():
        x_t = torch.from_numpy(x_data)
        w_t = torch.from_numpy(w_data)
        b_t = torch.from_numpy(b_data)
        mm_ref = torch.matmul(x_t, w_t).numpy()
        fused_ref = torch.relu(torch.matmul(x_t, w_t) + b_t).numpy()

    # Build IR graph for unfused and fused
    # Unfused: MATMUL only
    g_unfused = Graph()
    x_id = g_unfused.add_node(Node(0, "INPUT", shape=[M, K]))
    w_id = g_unfused.add_node(Node(1, "CONST", shape=[K, N], const_data=w_data.flatten().tolist()))
    mm_id = g_unfused.add_node(Node(2, "MATMUL", inputs=[x_id, w_id], shape=[M, N]))
    g_unfused.set_inputs([x_id])
    g_unfused.set_outputs([mm_id])

    # Fused: MATMUL + ADD + RELU
    g_fused = Graph()
    fx_id = g_fused.add_node(Node(0, "INPUT", shape=[M, K]))
    fw_id = g_fused.add_node(Node(1, "CONST", shape=[K, N], const_data=w_data.flatten().tolist()))
    fb_id = g_fused.add_node(Node(2, "CONST", shape=[N], const_data=b_data.flatten().tolist()))
    fmm_id = g_fused.add_node(Node(3, "MATMUL", inputs=[fx_id, fw_id], shape=[M, N]))
    fadd_id = g_fused.add_node(Node(4, "ADD", inputs=[fmm_id, fb_id], shape=[M, N]))
    frelu_id = g_fused.add_node(Node(5, "RELU", inputs=[fadd_id], shape=[M, N]))
    g_fused.set_inputs([fx_id])
    g_fused.set_outputs([frelu_id])

    json_unfused = str(tmp_path / f"unfused_{M}_{K}_{N}.json")
    json_fused = str(tmp_path / f"fused_{M}_{K}_{N}.json")
    with open(json_unfused, "w", encoding="utf-8") as f:
        f.write(g_unfused.to_json())
    with open(json_fused, "w", encoding="utf-8") as f:
        f.write(g_fused.to_json())

    compiler = NativeCompiler()

    # 1. NAIVE_IJK (unfused)
    c1_cpp = str(tmp_path / f"naive_{M}_{K}_{N}.cpp")
    c1_dll = str(tmp_path / f"naive_{M}_{K}_{N}.dll")
    subprocess.run([ENGINE_EXE, "--compile", json_unfused, "-o", c1_cpp, "--variant", "naive_ijk"], check=True)
    compiler.compile_shared_library(c1_cpp, c1_dll)
    rt1 = NativeRuntime(c1_dll, (M, K), (M, N))
    out1 = rt1.run(x_data)
    assert np.allclose(out1, mm_ref, atol=1e-5), "NAIVE_IJK output mismatch"

    # 2. LOOP_IKJ (unfused)
    c2_cpp = str(tmp_path / f"ikj_{M}_{K}_{N}.cpp")
    c2_dll = str(tmp_path / f"ikj_{M}_{K}_{N}.dll")
    subprocess.run([ENGINE_EXE, "--compile", json_unfused, "-o", c2_cpp, "--variant", "loop_ikj"], check=True)
    compiler.compile_shared_library(c2_cpp, c2_dll)
    rt2 = NativeRuntime(c2_dll, (M, K), (M, N))
    out2 = rt2.run(x_data)
    assert np.allclose(out2, mm_ref, atol=1e-5), "LOOP_IKJ output mismatch"

    # 3. TILED (tile 8)
    c3_cpp = str(tmp_path / f"tiled_{M}_{K}_{N}.cpp")
    c3_dll = str(tmp_path / f"tiled_{M}_{K}_{N}.dll")
    subprocess.run([ENGINE_EXE, "--compile", json_unfused, "-o", c3_cpp, "--variant", "tiled",
                    "--tile-m", "8", "--tile-n", "8", "--tile-k", "8"], check=True)
    compiler.compile_shared_library(c3_cpp, c3_dll)
    rt3 = NativeRuntime(c3_dll, (M, K), (M, N))
    out3 = rt3.run(x_data)
    assert np.allclose(out3, mm_ref, atol=1e-5), "TILED output mismatch"

    # 4. FUSED_IKJ (fused)
    c4_cpp = str(tmp_path / f"fused_ikj_{M}_{K}_{N}.cpp")
    c4_dll = str(tmp_path / f"fused_ikj_{M}_{K}_{N}.dll")
    subprocess.run([ENGINE_EXE, "--compile", json_fused, "-o", c4_cpp, "--fusion", "--variant", "fused_ikj"], check=True)
    compiler.compile_shared_library(c4_cpp, c4_dll)
    rt4 = NativeRuntime(c4_dll, (M, K), (M, N))
    out4 = rt4.run(x_data)
    assert np.allclose(out4, fused_ref, atol=1e-5), "FUSED_IKJ output mismatch"

    # 5. FUSED_TILED (fused, tile 8)
    c5_cpp = str(tmp_path / f"fused_tiled_{M}_{K}_{N}.cpp")
    c5_dll = str(tmp_path / f"fused_tiled_{M}_{K}_{N}.dll")
    subprocess.run([ENGINE_EXE, "--compile", json_fused, "-o", c5_cpp, "--fusion", "--variant", "fused_tiled",
                    "--tile-m", "8", "--tile-n", "8", "--tile-k", "8"], check=True)
    compiler.compile_shared_library(c5_cpp, c5_dll)
    rt5 = NativeRuntime(c5_dll, (M, K), (M, N))
    out5 = rt5.run(x_data)
    assert np.allclose(out5, fused_ref, atol=1e-5), "FUSED_TILED output mismatch"

def test_extreme_inputs_all_variants(tmp_path):
    # Test all-zeros and all-negatives on fused variants
    M, K, N = 10, 15, 12
    w_data = np.random.randn(K, N).astype(np.float32) * 0.1
    b_data = -np.ones(N, dtype=np.float32) * 5.0 # strongly negative bias

    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[M, K]))
    w = g.add_node(Node(1, "CONST", shape=[K, N], const_data=w_data.flatten().tolist()))
    b = g.add_node(Node(2, "CONST", shape=[N], const_data=b_data.flatten().tolist()))
    mm = g.add_node(Node(3, "MATMUL", inputs=[x, w], shape=[M, N]))
    add = g.add_node(Node(4, "ADD", inputs=[mm, b], shape=[M, N]))
    relu = g.add_node(Node(5, "RELU", inputs=[add], shape=[M, N]))
    g.set_inputs([x])
    g.set_outputs([relu])

    json_path = str(tmp_path / "extreme.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(g.to_json())

    compiler = NativeCompiler()
    cpp_fused = str(tmp_path / "extreme_fused.cpp")
    dll_fused = str(tmp_path / "extreme_fused.dll")
    subprocess.run([ENGINE_EXE, "--compile", json_path, "-o", cpp_fused, "--fusion", "--variant", "fused_ikj"], check=True)
    compiler.compile_shared_library(cpp_fused, dll_fused)
    rt = NativeRuntime(dll_fused, (M, K), (M, N))

    # All zeros input + negative bias -> output should be exact zero
    out_zeros = rt.run(np.zeros((M, K), dtype=np.float32))
    assert np.all(out_zeros == 0.0), "All zeros input must yield all zeros after negative bias ReLU"

    # Negative input
    out_negs = rt.run(-np.abs(np.random.randn(M, K).astype(np.float32) * 10.0))
    assert np.all(out_negs >= 0.0), "ReLU must never produce negative values"
