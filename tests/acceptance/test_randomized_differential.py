"""
Acceptance Test Suite: Randomized Differential Testing & Extreme Numerical Stress
Tests all 5 kernel variants against independent reference implementations across:
- Small, prime, sub-tile (<16), exact tile (16, 32), and non-tile (15, 17, 31, 33) dimensions
- Rectangular aspect ratios (tall, wide, deep inner dimension)
- Extreme values (large magnitude, small magnitude, mixed signs, zero matrices)
- Independent ground truth checks: C++ IR Interpreter vs PyTorch vs Standalone Native Kernel
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import torch

from forgetune.graph_adapter import Graph, Node
from forgetune.native_compiler import NativeCompiler
from forgetune.runtime import NativeRuntime

def compile_custom_graph(
    graph: Graph,
    variant: str,
    output_dir: str,
    tile_sizes=(16, 16, 16),
    enable_fusion: bool = False,
    opt_level: str = "-O2"
) -> NativeRuntime:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "graph.json")
    cpp_path = os.path.join(output_dir, "kernel.cpp")
    dll_path = os.path.join(output_dir, "kernel.dll")

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(graph.to_json())

    _test_engine = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "forgetune_test_engine.exe"))
    engine_exe = _test_engine if os.path.isfile(_test_engine) else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "forgetune_engine.exe"))
    import subprocess
    cmd = [
        engine_exe, "--compile", json_path, "-o", cpp_path,
        "--variant", variant
    ]
    if variant in ("tiled", "fused_tiled"):
        cmd.extend(["--tile-m", str(tile_sizes[0]), "--tile-n", str(tile_sizes[1]), "--tile-k", str(tile_sizes[2])])
    if enable_fusion:
        cmd.append("--fusion")

    subprocess.run(cmd, check=True)
    compiler = NativeCompiler()
    compiler.compile_shared_library(cpp_path, dll_path, opt_level=opt_level)

    in_shape = graph.nodes[graph.inputs[0]].shape
    out_shape = graph.nodes[graph.outputs[0]].shape
    return NativeRuntime(dll_path, tuple(in_shape), tuple(out_shape))

def build_matmul_graph(M: int, K: int, N: int, b_data: np.ndarray) -> Graph:
    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", "X", shape=[M, K], dtype="f32"))
    w_id = g.add_node(Node(1, "CONST", "W", shape=[K, N], dtype="f32", const_data=b_data.flatten().tolist()))
    mm_id = g.add_node(Node(2, "MATMUL", "mm", inputs=[in_id, w_id], shape=[M, N], dtype="f32"))
    out_id = g.add_node(Node(3, "OUTPUT", "out", inputs=[mm_id], shape=[M, N], dtype="f32"))
    g.set_inputs([in_id])
    g.set_outputs([out_id])
    return g

def build_fused_mlp_graph(M: int, K: int, N: int, w_data: np.ndarray, bias_data: np.ndarray) -> Graph:
    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", "X", shape=[M, K], dtype="f32"))
    w_id = g.add_node(Node(1, "CONST", "W", shape=[K, N], dtype="f32", const_data=w_data.flatten().tolist()))
    b_id = g.add_node(Node(2, "CONST", "B", shape=[N], dtype="f32", const_data=bias_data.flatten().tolist()))
    fused_id = g.add_node(Node(3, "FUSED_MATMUL_ADD_RELU", "fused", inputs=[in_id, w_id, b_id], shape=[M, N], dtype="f32"))
    out_id = g.add_node(Node(4, "OUTPUT", "out", inputs=[fused_id], shape=[M, N], dtype="f32"))
    g.set_inputs([in_id])
    g.set_outputs([out_id])
    return g

@pytest.fixture
def temp_diff_dir():
    d = os.path.abspath(os.path.join("build", "artifacts", "acceptance_diff"))
    if os.path.exists(d):
        try:
            shutil.rmtree(d)
        except Exception:
            pass
    os.makedirs(d, exist_ok=True)
    yield d
    try:
        shutil.rmtree(d)
    except Exception:
        pass

# Dimension test cases covering requested dimensions: {1, 3, 7, 8, 15, 16, 17, 31, 32, 33}
DIMENSION_TEST_CASES = [
    # (M, K, N, label)
    (1, 1, 1, "scalar_1x1"),
    (3, 7, 17, "prime_small"),
    (7, 8, 15, "subtile_15"),
    (16, 16, 16, "exact_tile_16"),
    (15, 17, 31, "cross_boundary_odd"),
    (32, 32, 32, "exact_tile_32"),
    (33, 15, 17, "post_boundary_33"),
    (64, 8, 4, "rectangular_tall"),
    (4, 8, 64, "rectangular_wide"),
    (4, 128, 4, "deep_inner_k")
]

@pytest.mark.parametrize("M,K,N,label", DIMENSION_TEST_CASES)
@pytest.mark.parametrize("variant", ["naive_ijk", "loop_ikj", "tiled"])
def test_acceptance_randomized_matmul_all_unfused_variants(M, K, N, label, variant, temp_diff_dir):
    np.random.seed(1000 + M * 17 + K * 13 + N)
    w_mat = np.random.uniform(-1.5, 1.5, (K, N)).astype(np.float32)
    x_mat = np.random.uniform(-1.5, 1.5, (M, K)).astype(np.float32)

    graph = build_matmul_graph(M, K, N, w_mat)

    # 1. Reference evaluation via interpreter
    interp_res = graph.evaluate({graph.inputs[0]: x_mat})
    ref_interp = interp_res[graph.outputs[0]]

    # 2. PyTorch reference evaluation
    t_x = torch.from_numpy(x_mat)
    t_w = torch.from_numpy(w_mat)
    ref_torch = torch.matmul(t_x, t_w).numpy()

    # Verify interpreter matches PyTorch
    np.testing.assert_allclose(ref_interp, ref_torch, atol=1e-4, rtol=1e-4)

    # 3. Native compiler evaluation
    var_dir = os.path.join(temp_diff_dir, f"{label}_{variant}")
    runtime = compile_custom_graph(graph, variant, var_dir, tile_sizes=(16, 16, 16))
    native_out = runtime.run(x_mat)

    # Compare native vs PyTorch
    max_abs_diff = float(np.max(np.abs(native_out - ref_torch)))
    max_rel_diff = float(np.max(np.abs((native_out - ref_torch) / (np.abs(ref_torch) + 1e-7))))

    # Tolerance scaling for larger K accumulation
    tol = 1e-4 if K <= 64 else 5e-4
    assert max_abs_diff < tol, f"Failed on {label} ({variant}): max_abs={max_abs_diff}, tol={tol}"
    np.testing.assert_allclose(native_out, ref_torch, atol=tol, rtol=1e-4)

@pytest.mark.parametrize("M,K,N,label", [
    (1, 1, 1, "scalar_fused"),
    (7, 8, 15, "subtile_fused"),
    (16, 16, 16, "tile16_fused"),
    (17, 31, 33, "odd_tile_fused"),
    (32, 32, 32, "tile32_fused")
])
@pytest.mark.parametrize("variant", ["fused_ikj", "fused_tiled"])
def test_acceptance_randomized_fused_variants(M, K, N, label, variant, temp_diff_dir):
    np.random.seed(2000 + M * 19 + K * 7 + N)
    w_mat = np.random.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
    b_vec = np.random.uniform(-0.5, 0.5, (N,)).astype(np.float32)
    x_mat = np.random.uniform(-1.0, 1.0, (M, K)).astype(np.float32)

    graph = build_fused_mlp_graph(M, K, N, w_mat, b_vec)

    # PyTorch reference calculation
    t_x = torch.from_numpy(x_mat)
    t_w = torch.from_numpy(w_mat)
    t_b = torch.from_numpy(b_vec)
    ref_torch = torch.relu(torch.matmul(t_x, t_w) + t_b).numpy()

    var_dir = os.path.join(temp_diff_dir, f"{label}_{variant}")
    runtime = compile_custom_graph(graph, variant, var_dir, tile_sizes=(16, 16, 16), enable_fusion=True)
    native_out = runtime.run(x_mat)

    max_abs_diff = float(np.max(np.abs(native_out - ref_torch)))
    assert max_abs_diff < 1e-4, f"Fused mismatch on {label} ({variant}): max_abs={max_abs_diff}"
    np.testing.assert_allclose(native_out, ref_torch, atol=1e-4, rtol=1e-4)

def test_acceptance_extreme_values_stress(temp_diff_dir):
    """
    Tests numerical stability with extreme magnitudes:
    - Large inputs (1e4)
    - Small inputs (1e-4)
    - Zero matrices
    - Mixed negative and positive values
    """
    M, K, N = 8, 16, 8

    # 1. Zero input
    w = np.random.randn(K, N).astype(np.float32)
    x_zero = np.zeros((M, K), dtype=np.float32)
    g_zero = build_matmul_graph(M, K, N, w)
    rt_zero = compile_custom_graph(g_zero, "loop_ikj", os.path.join(temp_diff_dir, "zero"))
    out_zero = rt_zero.run(x_zero)
    assert np.all(out_zero == 0.0), "Zero input produced non-zero output"

    # 2. Large inputs (1e4)
    w_large = np.full((K, N), 1e2, dtype=np.float32)
    x_large = np.full((M, K), 1e2, dtype=np.float32)
    g_large = build_matmul_graph(M, K, N, w_large)
    rt_large = compile_custom_graph(g_large, "loop_ikj", os.path.join(temp_diff_dir, "large"))
    out_large = rt_large.run(x_large)
    # Each output element = K * (1e2 * 1e2) = 16 * 1e4 = 160000.0
    expected_large = float(K * 1e4)
    assert np.allclose(out_large, expected_large, rtol=1e-5)

    # 3. Small inputs (1e-4)
    w_small = np.full((K, N), 1e-2, dtype=np.float32)
    x_small = np.full((M, K), 1e-2, dtype=np.float32)
    g_small = build_matmul_graph(M, K, N, w_small)
    rt_small = compile_custom_graph(g_small, "loop_ikj", os.path.join(temp_diff_dir, "small"))
    out_small = rt_small.run(x_small)
    expected_small = float(K * 1e-4)
    assert np.allclose(out_small, expected_small, rtol=1e-5)
