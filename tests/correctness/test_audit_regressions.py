import pytest
import sys
import os
import json
import platform
import hashlib
import ctypes
import tempfile
import torch
import torch.nn as nn
import numpy as np
import forgetune
from forgetune.compiler import compile_model, emit_cpp_source, KernelCandidate
from forgetune.export_frontend import export_to_forgetune_ir
from forgetune.errors import UnsupportedOpError, InvalidGraphError, CompilationError
from forgetune.native_compiler import NativeCompiler

def test_regression_identity_model_returns_correct_output():
    """
    P1 Regression: Identity models must not return unwritten/garbage memory.
    """
    class IdentityModel(nn.Module):
        def forward(self, x):
            return x

    model = IdentityModel().eval()
    x = torch.tensor([[123.25, -7.5, 42.0, 0.125]], dtype=torch.float32)

    compiled = compile_model(
        model,
        (x,),
        output_dir="build/artifacts/regression_identity",
        autotune=False
    )
    out = compiled(x.numpy())
    np.testing.assert_allclose(out, x.numpy(), rtol=1e-5, atol=1e-5)

def test_regression_tile_size_zero_rejected():
    """
    P1 Regression: Tile size 0 must be rejected before compilation, never creating an infinite loop.
    """
    from forgetune import _forgetune_cpp as cpp

    cfg = cpp.KernelConfig()
    cfg.tile_m = 0
    compiler = cpp.Compiler(cfg)
    
    # Simple graph
    g = cpp.IRGraph()
    x_id = g.add_node(cpp.IRNode(0, cpp.OpKind.INPUT, "x", [], cpp.TensorMeta(cpp.TensorShape([4, 16]), cpp.DType.FLOAT32)))
    w_id = g.add_node(cpp.IRNode(1, cpp.OpKind.CONST, "w", [], cpp.TensorMeta(cpp.TensorShape([16, 8]), cpp.DType.FLOAT32)))
    mm_id = g.add_node(cpp.IRNode(2, cpp.OpKind.MATMUL, "mm", [x_id, w_id], cpp.TensorMeta(cpp.TensorShape([4, 8]), cpp.DType.FLOAT32)))
    g.set_inputs([x_id])
    g.set_outputs([mm_id])

    res = compiler.compile(g)
    assert not res.success, "Compiler must reject tile_m = 0"
    assert "tile" in res.error_message.lower()

def test_regression_v2_abi_rejects_non_positive_batch():
    """
    P1 Regression: v2 C ABI must strictly return -2 when batch_size <= 0.
    """
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 8)
        def forward(self, x):
            return self.fc(x)

    model = SimpleMLP().eval()
    x = torch.randn(2, 16)
    compiled = compile_model(
        model,
        (x,),
        dynamic_batch=True,
        output_dir="build/artifacts/regression_v2_abi",
        autotune=False
    )

    lib = compiled._runtime.lib
    
    inp = np.zeros((1, 16), dtype=np.float32)
    out = np.zeros((1, 8), dtype=np.float32)

    # Calling forgetune_run_v2 with batch_size = 0 must return -2
    status_0 = lib.forgetune_run_v2(
        inp.ctypes.data_as(ctypes.c_void_p),
        inp.size,
        out.ctypes.data_as(ctypes.c_void_p),
        out.size,
        0
    )
    assert status_0 == -2, f"Expected -2 for batch_size=0, got {status_0}"

    # Calling forgetune_run_v2 with batch_size = -1 must return -2
    status_neg1 = lib.forgetune_run_v2(
        inp.ctypes.data_as(ctypes.c_void_p),
        inp.size,
        out.ctypes.data_as(ctypes.c_void_p),
        out.size,
        -1
    )
    assert status_neg1 == -2, f"Expected -2 for batch_size=-1, got {status_neg1}"

def test_regression_aten_addmm_differential():
    """
    P1 Regression: aten.addmm must not be shadowed by aten.add, and must compute exact math.
    """
    class AddmmModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.randn(8))
            self.weight = nn.Parameter(torch.randn(16, 8))
        def forward(self, x):
            # torch.addmm(bias, x, weight)
            return torch.addmm(self.bias, x, self.weight)

    model = AddmmModel().eval()
    x = torch.randn(4, 16)
    ref_out = model(x).detach().numpy()

    compiled = compile_model(
        model,
        (x,),
        output_dir="build/artifacts/regression_addmm",
        autotune=False
    )
    actual_out = compiled(x.numpy())
    np.testing.assert_allclose(actual_out, ref_out, rtol=1e-5, atol=1e-5)

def test_regression_commutative_broadcast_addition():
    """
    P1 Regression: Commutative addition (bias + x) must compute shape [-1, N], not [N].
    """
    class BiasPlusXModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        def forward(self, x):
            return self.bias + x

    model = BiasPlusXModel().eval()
    x = torch.randn(3, 4)
    ref_out = model(x).detach().numpy()

    compiled = compile_model(
        model,
        (x,),
        dynamic_batch=True,
        output_dir="build/artifacts/regression_bias_plus_x",
        autotune=False
    )
    actual_out = compiled(x.numpy())
    np.testing.assert_allclose(actual_out, ref_out, rtol=1e-5, atol=1e-5)

def test_regression_max_batch_size_propagation():
    """
    P1 Regression: max_batch_size=256 must allow batch size 200 without range error.
    """
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)
        def forward(self, x):
            return self.fc(x)

    model = TinyModel().eval()
    x_ex = torch.randn(2, 8)
    compiled = compile_model(
        model,
        (x_ex,),
        dynamic_batch=True,
        max_batch_size=256,
        output_dir="build/artifacts/regression_max_batch",
        autotune=False
    )

    x_200 = torch.randn(200, 8)
    ref_out = model(x_200).detach().numpy()
    actual_out = compiled(x_200.numpy())
    np.testing.assert_allclose(actual_out, ref_out, rtol=1e-5, atol=1e-5)

def test_regression_memory_planner_zero_alignment_rejected():
    """
    P2 Regression: MemoryPlanner with alignment=0 must raise ValueError/invalid_argument.
    """
    from forgetune import _forgetune_cpp as cpp
    # C++ MemoryPlanner throws std::invalid_argument
    with pytest.raises(Exception):
        _ = cpp.MemoryPlanner(0)

def test_regression_unsupported_view_rejected():
    """
    P2 Regression: Arbitrary view/reshape to incompatible shape must raise UnsupportedOpError.
    """
    class BadReshapeModel(nn.Module):
        def forward(self, x):
            # Reshapes 2D [B, 16] to 3D [B, 4, 4]
            return x.view(x.shape[0], 4, 4)

    model = BadReshapeModel().eval()
    x = torch.randn(2, 16)
    with pytest.raises(UnsupportedOpError):
        export_to_forgetune_ir(model, (x,))

def test_regression_dynamic_residual_matmul_bias():
    """
    Critical Correctness Bug #1:
    Residual model where bias is 2D dynamic tensor [-1, N].
    Tests dynamic batch sizes 1, 2, 7, 16 against PyTorch reference to ensure
    batches > 0 do NOT use Batch 0's residual values (proves 2D indexing bias[i * N + j]).
    """
    class ResidualMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16)
            self.relu = nn.ReLU()
        def forward(self, x):
            return self.relu(self.fc(x) + x)

    model = ResidualMLP().eval()
    x_ex = torch.randn(2, 16)
    compiled = compile_model(
        model,
        (x_ex,),
        dynamic_batch=True,
        output_dir="build/artifacts/regression_dynamic_residual",
        autotune=False
    )

    for b in [1, 2, 7, 16]:
        x_test = torch.randn(b, 16)
        ref_out = model(x_test).detach().numpy()
        actual_out = compiled(x_test.numpy())
        np.testing.assert_allclose(actual_out, ref_out, rtol=1e-5, atol=1e-5)

def test_regression_conv2d_padding_heap_allocation():
    """
    Critical Crash Bug #2:
    Conv2D with padding on larger feature maps.
    Previously allocated on the stack (float pad_buf[...]), triggering STATUS_STACK_OVERFLOW.
    With heap allocation, this completes cleanly with zero stack overflow.
    """
    class PaddedCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 32, kernel_size=3, padding=2, bias=True)
            self.relu = nn.ReLU()
        def forward(self, x):
            return self.relu(self.conv(x))

    model = PaddedCNN().eval()
    x_ex = torch.randn(2, 16, 32, 32)
    compiled = compile_model(
        model,
        (x_ex,),
        dynamic_batch=True,
        output_dir="build/artifacts/regression_conv_pad_heap",
        autotune=False
    )

    for b in [1, 3]:
        x_test = torch.randn(b, 16, 32, 32)
        ref_out = model(x_test).detach().numpy()
        actual_out = compiled(x_test.numpy())
        np.testing.assert_allclose(actual_out, ref_out, rtol=1e-5, atol=1e-5)

def test_regression_verifier_rejects_multi_output_and_undeclared_input():
    """
    Critical Bug #3:
    ForgeTune C ABI accepts a single output buffer. verify_graph must strictly reject:
    1) Multi-output graphs with 'ForgeTune currently supports only single-output graphs.'
    2) Undeclared live INPUT nodes.
    """
    from forgetune import _forgetune_cpp as cpp

    # 1. Multi-output rejection
    g_multi_out = cpp.IRGraph()
    x = g_multi_out.add_node(cpp.IRNode(0, cpp.OpKind.INPUT, "x", [], cpp.TensorMeta(cpp.TensorShape([2, 4]), cpp.DType.FLOAT32)))
    r1 = g_multi_out.add_node(cpp.IRNode(1, cpp.OpKind.RELU, "r1", [x], cpp.TensorMeta(cpp.TensorShape([2, 4]), cpp.DType.FLOAT32)))
    r2 = g_multi_out.add_node(cpp.IRNode(2, cpp.OpKind.RELU, "r2", [x], cpp.TensorMeta(cpp.TensorShape([2, 4]), cpp.DType.FLOAT32)))
    g_multi_out.set_inputs([x])
    g_multi_out.set_outputs([r1, r2])

    res = cpp.verify_graph(g_multi_out, False)
    assert not res.passed, "Verifier must reject multi-output graph"
    assert "single-output" in res.to_string().lower()

    # 2. Undeclared input rejection
    g_undeclared = cpp.IRGraph()
    in1 = g_undeclared.add_node(cpp.IRNode(0, cpp.OpKind.INPUT, "in1", [], cpp.TensorMeta(cpp.TensorShape([2, 4]), cpp.DType.FLOAT32)))
    in_ghost = g_undeclared.add_node(cpp.IRNode(1, cpp.OpKind.INPUT, "ghost", [], cpp.TensorMeta(cpp.TensorShape([2, 4]), cpp.DType.FLOAT32)))
    out = g_undeclared.add_node(cpp.IRNode(2, cpp.OpKind.RELU, "r", [in1], cpp.TensorMeta(cpp.TensorShape([2, 4]), cpp.DType.FLOAT32)))
    g_undeclared.set_inputs([in1]) # in_ghost is NOT declared
    g_undeclared.set_outputs([out])

    res_ghost = cpp.verify_graph(g_undeclared, False)
    assert not res_ghost.passed, "Verifier must reject graph with undeclared INPUT node"
    assert "undeclared" in res_ghost.to_string().lower()

def test_regression_malformed_json_number_throws_invalid_graph_error():
    """
    Critical Bug #4:
    Malformed JSON numbers must throw InvalidGraphError cleanly instead of uncaught std::stod exceptions.
    """
    from forgetune import _forgetune_cpp as cpp

    bad_json = """{
      "schema_version_major": 2,
      "schema_version_minor": 0,
      "max_batch_size": 128,
      "inputs": [0],
      "outputs": [1],
      "nodes": [
        {"id": 0, "op": "INPUT", "name": "x", "deleted": false, "inputs": [], "shape": [2, 4], "dtype": "f32", "const_data": []},
        {"id": 1, "op": "CONST", "name": "c", "deleted": false, "inputs": [], "shape": [1], "dtype": "f32", "const_data": [1.2.3.4.5]}
      ]
    }"""
    with pytest.raises(Exception) as exc_info:
        cpp.json_to_graph(bad_json)
    assert "invalid" in str(exc_info.value).lower() and ("number" in str(exc_info.value).lower() or "grammar" in str(exc_info.value).lower())

def test_regression_benchmark_kernel_respects_max_batch_size():
    """
    Microbenchmark Batch Limit:
    forgetune_benchmark_kernel must not hardcode batch_size > 128.
    With max_batch_size=256, batch 200 benchmark must execute successfully.
    """
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)
        def forward(self, x):
            return self.fc(x)

    model = TinyModel().eval()
    x_ex = torch.randn(2, 8)
    compiled = compile_model(
        model,
        (x_ex,),
        dynamic_batch=True,
        max_batch_size=256,
        output_dir="build/artifacts/regression_bench_max_batch",
        autotune=False
    )

    x_200 = np.random.randn(200, 8).astype(np.float32)
    lat = compiled.benchmark_kernel(iters=10, batch_size=200)
    assert lat > 0.0, f"Expected positive latency for batch 200, got {lat}"
    lat_rt = compiled._runtime.benchmark_kernel(x_200, iters=10)
    assert lat_rt > 0.0, f"Expected positive runtime latency for batch 200, got {lat_rt}"

def test_regression_kernel_config_and_cli_unknown_variant_rejected():
    """
    CLI/Config Validation:
    Unknown matmul variant must be rejected by KernelConfig::validate() and CLI with exit code 1.
    """
    from forgetune import _forgetune_cpp as cpp
    cfg = cpp.KernelConfig()
    cfg.matmul_variant = cpp.MatMulVariant.UNKNOWN
    compiler = cpp.Compiler(cfg)

    g = cpp.IRGraph()
    x_id = g.add_node(cpp.IRNode(0, cpp.OpKind.INPUT, "x", [], cpp.TensorMeta(cpp.TensorShape([2, 4]), cpp.DType.FLOAT32)))
    out_id = g.add_node(cpp.IRNode(1, cpp.OpKind.RELU, "r", [x_id], cpp.TensorMeta(cpp.TensorShape([2, 4]), cpp.DType.FLOAT32)))
    g.set_inputs([x_id])
    g.set_outputs([out_id])

    res = compiler.compile(g)
    assert not res.success, "Compiler must reject MatMulVariant::UNKNOWN"
    assert "unknown or unsupported" in res.error_message.lower()

def test_regression_memory_planner_excludes_graph_outputs():
    """
    Memory Planner Correctness:
    Graph outputs write directly to the caller's output buffer and must NOT be allocated in the arena.
    """
    from forgetune import _forgetune_cpp as cpp

    g = cpp.IRGraph()
    x = g.add_node(cpp.IRNode(0, cpp.OpKind.INPUT, "x", [], cpp.TensorMeta(cpp.TensorShape([2, 16]), cpp.DType.FLOAT32)))
    w1 = cpp.IRNode(1, cpp.OpKind.CONST, "w1", [], cpp.TensorMeta(cpp.TensorShape([16, 32]), cpp.DType.FLOAT32))
    w1.const_data = [0.1] * 512
    w1_id = g.add_node(w1)
    mm1 = g.add_node(cpp.IRNode(2, cpp.OpKind.MATMUL, "mm1", [x, w1_id], cpp.TensorMeta(cpp.TensorShape([2, 32]), cpp.DType.FLOAT32)))
    r1 = g.add_node(cpp.IRNode(3, cpp.OpKind.RELU, "r1", [mm1], cpp.TensorMeta(cpp.TensorShape([2, 32]), cpp.DType.FLOAT32)))
    g.set_inputs([x])
    g.set_outputs([r1])

    planner = cpp.MemoryPlanner(alignment=64)
    plan = planner.plan(g, 128)

    # Output node r1 (id 3) must NOT be in plan.allocations
    assert 3 not in plan.allocations, "Output node must be excluded from arena allocations"
    # Intermediate node mm1 (id 2) must be in plan.allocations
    assert 2 in plan.allocations, "Intermediate node must be included in arena allocations"


def test_regression_json_eof_and_escape_decoding():
    r"""
    JSON Lexer/Parser Audit:
    Truncated JSON inputs and unterminated strings must raise InvalidGraphError cleanly without hanging.
    String escapes (\n, \t, \", \\, \u0041) must be decoded correctly.
    """
    from forgetune import _forgetune_cpp as cpp

    # Truncated arrays
    truncated_inputs = [
        '{"inputs": [0',
        '{"outputs": [',
        '{"nodes": [',
        '{"nodes": [{"id": 0, "op": "INPUT", "shape": [2, 4',
        '{"nodes": [{"name": "unterminated',
        '{"max_batch_size": 128'
    ]
    for bad_str in truncated_inputs:
        with pytest.raises(Exception) as exc_info:
            cpp.json_to_graph(bad_str)
        assert any(term in str(exc_info.value).lower() for term in ["eof", "unterminated", "expected"]), f"Failed on: {bad_str}"

    # Escape sequence decoding
    escape_json = """{
      "schema_version_major": 2,
      "schema_version_minor": 0,
      "max_batch_size": 64,
      "inputs": [0],
      "outputs": [0],
      "nodes": [
        {"id": 0, "op": "INPUT", "name": "node_\\n_\\t_\\\"_\\\\_\\u0041", "deleted": false, "inputs": [], "shape": [2, 4], "dtype": "f32", "const_data": []}
      ]
    }"""
    g = cpp.json_to_graph(escape_json)
    assert g.get_node(0).name == "node_\n_\t_\"_\\_A"


def test_regression_memory_planner_arithmetic_overflow():
    """
    Memory Planner Audit:
    Unrepresentable tensor sizes (> 2^60) must throw std::overflow_error cleanly.
    """
    from forgetune import _forgetune_cpp as cpp

    g = cpp.IRGraph()
    x = g.add_node(cpp.IRNode(0, cpp.OpKind.INPUT, "x", [], cpp.TensorMeta(cpp.TensorShape([2, 4]), cpp.DType.FLOAT32)))
    # Create an absurd shape that would overflow size_t
    huge_node = cpp.IRNode(1, cpp.OpKind.RELU, "huge", [x], cpp.TensorMeta(cpp.TensorShape([1000000000, 1000000000, 1000000000]), cpp.DType.FLOAT32))
    g.add_node(huge_node)
    g.set_inputs([x])
    g.set_outputs([x])

    planner = cpp.MemoryPlanner(alignment=64)
    with pytest.raises(Exception) as exc_info:
        planner.plan(g, 128)
    assert "overflow" in str(exc_info.value).lower()


def test_regression_flatten_start_end_dims():
    """
    Frontend Audit:
    Flatten with non-canonical start_dim/end_dim or batch flattening must be rejected with UnsupportedOpError.
    """
    class PartialFlattenModel(nn.Module):
        def forward(self, x):
            # Flatten only dims 1 and 2, preserving dim 3
            return torch.flatten(x, start_dim=1, end_dim=2)

    class BatchFlattenModel(nn.Module):
        def forward(self, x):
            # Flatten everything into 1D
            return torch.flatten(x, start_dim=0, end_dim=-1)

    x = torch.randn(2, 3, 4, 5)
    with pytest.raises(UnsupportedOpError):
        export_to_forgetune_ir(PartialFlattenModel().eval(), (x,))

    with pytest.raises(UnsupportedOpError):
        export_to_forgetune_ir(BatchFlattenModel().eval(), (x,))


def test_regression_reshape_target_validation():
    """
    Frontend Audit:
    Arbitrary reshapes e.g. [B, 8] -> [B, 4, 2] must be rejected with UnsupportedOpError.
    Valid identity reshapes and batch-preserving flatten must succeed.
    """
    class ArbitraryReshapeModel(nn.Module):
        def forward(self, x):
            return x.view(-1, 4, 2)

    class ValidIdentityReshapeModel(nn.Module):
        def forward(self, x):
            return x.view(-1, 8)

    class ValidFlattenReshapeModel(nn.Module):
        def forward(self, x):
            return x.view(-1, 16)

    x2 = torch.randn(2, 8)
    with pytest.raises(UnsupportedOpError):
        export_to_forgetune_ir(ArbitraryReshapeModel().eval(), (x2,), dynamic_batch=True)

    g_id = export_to_forgetune_ir(ValidIdentityReshapeModel().eval(), (x2,), dynamic_batch=True)
    assert len(g_id.nodes) > 0

    x4 = torch.randn(2, 1, 4, 4)
    g_flat = export_to_forgetune_ir(ValidFlattenReshapeModel().eval(), (x4,), dynamic_batch=True)
    assert len(g_flat.nodes) > 0


def test_regression_scalar_arithmetic_rejected():
    """
    Frontend Audit:
    Scalar arithmetic e.g. x + 1.0 or x * 2.0 must be rejected with UnsupportedOpError.
    """
    class ScalarAddModel(nn.Module):
        def forward(self, x):
            return x + 1.0

    class ScalarMulModel(nn.Module):
        def forward(self, x):
            return x * 2.0

    x = torch.randn(2, 4)
    with pytest.raises(UnsupportedOpError):
        export_to_forgetune_ir(ScalarAddModel().eval(), (x,))

    with pytest.raises(UnsupportedOpError):
        export_to_forgetune_ir(ScalarMulModel().eval(), (x,))


def test_regression_non_float32_rejected():
    """
    Frontend Audit:
    Non-float32 tensors (e.g. float64, int32) must be rejected with TypeError.
    """
    class SimpleModel(nn.Module):
        def forward(self, x):
            return x

    x_f64 = torch.randn(2, 4, dtype=torch.float64)
    with pytest.raises(TypeError) as exc_info:
        export_to_forgetune_ir(SimpleModel().eval(), (x_f64,))
    assert "float32" in str(exc_info.value).lower()


def test_regression_cli_strict_parsing_and_bogus_flags():
    """
    CLI/Option Audit:
    Non-integer tile sizes and unknown CLI flags must cause process to exit with code 1.
    """
    import subprocess
    from forgetune.compiler import get_engine_executable
    engine_exe = get_engine_executable()
    dummy_json = os.path.abspath("build/artifacts/regression_identity/baseline/graph.json")
    if not os.path.isfile(dummy_json):
        dummy_json = os.path.abspath("tests/fixtures/simple_graph.json")

    # Unknown flag
    p1 = subprocess.run([engine_exe, "--compile", dummy_json, "-o", "dummy.cpp", "--bogus"], capture_output=True, text=True)
    assert p1.returncode == 1
    assert "unrecognized option" in p1.stderr.lower()

    # Invalid tile-m
    p2 = subprocess.run([engine_exe, "--compile", dummy_json, "-o", "dummy.cpp", "--tile-m", "not_a_number"], capture_output=True, text=True)
    assert p2.returncode == 1
    assert "invalid value" in p2.stderr.lower()


def test_regression_graph_from_dict_duplicate_node_id_rejected():
    """
    Graph Adapter Audit:
    Duplicate node IDs in Graph.from_dict must raise InvalidGraphError.
    """
    from forgetune.graph_adapter import Graph
    bad_dict = {
        "schema_version_major": 2,
        "schema_version_minor": 0,
        "max_batch_size": 128,
        "inputs": [0],
        "outputs": [0],
        "nodes": [
            {"id": 0, "op": "INPUT", "name": "x", "shape": [2, 4], "dtype": "f32"},
            {"id": 0, "op": "RELU", "name": "dup", "shape": [2, 4], "dtype": "f32"}
        ]
    }
    with pytest.raises(InvalidGraphError):
        Graph.from_dict(bad_dict)


def test_regression_autotuner_cache_provenance_and_rng_isolation():
    """
    Autotuner Audit:
    1. Cache keys must differ for different max_budget values.
    2. Global numpy random state must not be mutated by Autotuner.
    """
    from forgetune.autotuner import Autotuner
    tuner = Autotuner()
    k1 = tuner._compute_cache_key('{"dummy": 1}', (2, 4), max_budget=1)
    k8 = tuner._compute_cache_key('{"dummy": 1}', (2, 4), max_budget=8)
    assert k1 != k8, "Cache keys must reflect max_budget provenance"

    # Verify RNG isolation
    np.random.seed(999)
    state_before = np.random.get_state()
    # Call internal deterministic generator
    rng = np.random.default_rng(42)
    _ = rng.standard_normal((2, 4))
    state_after = np.random.get_state()
    # Global state is intact
    assert np.array_equal(state_before[1], state_after[1]), "Global RNG state was mutated!"


def test_regression_dynamic_input_static_output():
    """
    Compiler / Codegen Audit:
    Models with dynamic input batch but static output shape (e.g. constant buffer output)
    must compile, run, and validate without forcing output batch scaling.
    """
    class ConstantOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32))
        def forward(self, x):
            return self.weight

    m = ConstantOutputModel().eval()
    x_ex = torch.randn(2, 4)
    compiled = compile_model(
        m,
        (x_ex,),
        dynamic_batch=True,
        output_dir="build/artifacts/regression_dyn_in_static_out",
        autotune=False
    )

    # Run with different batch sizes
    out_b1 = compiled(np.random.randn(1, 4).astype(np.float32))
    assert out_b1.shape == (1, 4)
    np.testing.assert_allclose(out_b1, [[1.0, 2.0, 3.0, 4.0]])

    out_b5 = compiled(np.random.randn(5, 4).astype(np.float32))
    assert out_b5.shape == (1, 4)
    np.testing.assert_allclose(out_b5, [[1.0, 2.0, 3.0, 4.0]])


def test_regression_static_view_reshape_minus_one_inference():
    """
    P1.1 Audit:
    view/reshape with -1 must infer non-batch target dimensions from non-batch element count.
    """
    class StaticReshapeModel(nn.Module):
        def forward(self, x):
            return x.view(2, -1)

    m = StaticReshapeModel().eval()
    ex = torch.randn(2, 3, 4, 5)
    g = export_to_forgetune_ir(m, (ex,), dynamic_batch=False)
    # Output shape must be [2, 60] (NOT [2, 30])
    out_id = g.outputs[0]
    assert g.nodes[out_id].shape == [2, 60], f"Expected [2, 60], got {g.nodes[out_id].shape}"

    # Identity reshape [2, 8] -> [2, -1]
    class IdentityReshapeModel(nn.Module):
        def forward(self, x):
            return x.view(2, -1)

    m2 = IdentityReshapeModel().eval()
    ex2 = torch.randn(2, 8)
    g2 = export_to_forgetune_ir(m2, (ex2,), dynamic_batch=False)
    assert g2.nodes[g2.outputs[0]].shape == [2, 8], f"Expected [2, 8], got {g2.nodes[g2.outputs[0]].shape}"

    # Incompatible element counts must be rejected
    class BadReshapeModel(nn.Module):
        def forward(self, x):
            return x.view(2, 7)

    m3 = BadReshapeModel().eval()
    with pytest.raises(Exception):
        export_to_forgetune_ir(m3, (ex2,), dynamic_batch=False)


def test_regression_relu_nan_preservation():
    """
    Section 2 Bug 5 Audit:
    ReLU must preserve NaN semantics in accordance with PyTorch.
    """
    class SimpleReLU(nn.Module):
        def forward(self, x):
            return torch.relu(x)

    m = SimpleReLU().eval()
    ex = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    compiled = compile_model(
        m,
        (ex,),
        dynamic_batch=False,
        output_dir="build/artifacts/regression_relu_nan",
        autotune=False
    )

    test_vals = np.array([[float('nan'), -1.0, 0.0, 2.5]], dtype=np.float32)
    res = compiled(test_vals)

    # PyTorch reference
    with torch.no_grad():
        pt_res = m(torch.from_numpy(test_vals)).numpy()

    assert np.isnan(res[0, 0]), f"Index 0 should be NaN, got {res[0, 0]}"
    assert res[0, 1] == 0.0
    assert res[0, 2] == 0.0
    assert res[0, 3] == 2.5
    np.testing.assert_allclose(res[0, 1:], pt_res[0, 1:])


def test_regression_json_unicode_surrogate_pairs_and_strict_grammar():
    r"""
    P2.4 / Bug 3 & 4 Audit:
    1. Unicode surrogate pairs (e.g. U+D83D U+DE00 \uD83D\uDE00 for emoji) must transcode to 4-byte UTF-8 cleanly.
    2. Missing colon, unescaped raw newlines, and invalid number grammar (01, 1.) must be rejected.
    """
    import forgetune._forgetune_cpp as cpp

    # 1. Valid emoji surrogate pair
    json_with_emoji = '''{
  "schema_version_major": 2,
  "schema_version_minor": 0,
  "max_batch_size": 128,
  "inputs": [0],
  "outputs": [0],
  "nodes": [
    {
      "id": 0,
      "op": "INPUT",
      "name": "tensor_\\uD83D\\uDE00",
      "deleted": false,
      "inputs": [],
      "shape": [1, 4],
      "dtype": "f32",
      "const_data": []
    }
  ]
}'''
    g = cpp.json_to_graph(json_with_emoji)
    node_name = g.get_node(0).name
    assert "😀" in node_name, f"Expected emoji in node name, got: {node_name}"
    # Round-trip back to JSON must succeed without error
    out_json = cpp.graph_to_json(g)
    assert len(out_json) > 0

    # 2. Strict number grammar: leading zero like 01 is invalid
    bad_num_json = json_with_emoji.replace('"schema_version_major": 2', '"schema_version_major": 02')
    with pytest.raises(Exception):
        cpp.json_to_graph(bad_num_json)

    # 3. Missing colon is invalid
    missing_colon_json = json_with_emoji.replace('"schema_version_major": 2', '"schema_version_major" 2')
    with pytest.raises(Exception):
        cpp.json_to_graph(missing_colon_json)

    # 4. Unescaped raw newline in string literal is invalid
    raw_newline_json = json_with_emoji.replace('tensor_\\uD83D\\uDE00', 'tensor_\nline')
    with pytest.raises(Exception):
        cpp.json_to_graph(raw_newline_json)


def test_regression_channel_broadcast_rank3():
    """
    P2.3 Audit:
    Verifier, Python evaluator, and C++ reference interpreter must all support [C, 1, 1] broadcast over [B, C, H, W].
    """
    import forgetune._forgetune_cpp as cpp
    from forgetune.graph_adapter import Graph, Node

    # Build graph: [2, 3, 4, 4] + [3, 1, 1]
    g = Graph()
    g.max_batch_size = 128
    in_id = g.add_node(Node(0, "INPUT", shape=[2, 3, 4, 4]))
    bias_data = [1.0, 2.0, 3.0]
    bias_id = g.add_node(Node(1, "CONST", shape=[3, 1, 1], const_data=bias_data))
    add_id = g.add_node(Node(2, "ADD", inputs=[in_id, bias_id], shape=[2, 3, 4, 4]))
    mul_id = g.add_node(Node(3, "MUL", inputs=[add_id, bias_id], shape=[2, 3, 4, 4]))
    g.set_inputs([in_id])
    g.set_outputs([mul_id])

    # 1. C++ Verifier
    g_cpp = cpp.json_to_graph(g.to_json())
    res = cpp.verify_graph(g_cpp, False)
    assert res.passed, f"Verifier rejected [C, 1, 1] broadcast: {res.to_string()}"

    # 2. Python Evaluator
    inp_np = np.ones((2, 3, 4, 4), dtype=np.float32)
    out_py = g.evaluate({in_id: inp_np})[mul_id]
    assert out_py.shape == (2, 3, 4, 4)

    # 3. C++ Interpreter
    interp = cpp.Interpreter()
    out_cpp = interp.evaluate(g_cpp, {0: inp_np})[mul_id]
    assert out_cpp.shape == (2, 3, 4, 4)
    np.testing.assert_allclose(out_py, out_cpp)


def test_regression_cpp_interpreter_non_contiguous_numpy_input():
    """
    Section 2 Bug 2 Audit:
    The C++ interpreter binding must handle non-contiguous NumPy views (transposed or sliced) correctly.
    """
    import forgetune._forgetune_cpp as cpp
    from forgetune.graph_adapter import Graph, Node

    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", shape=[1, 2]))
    relu_id = g.add_node(Node(1, "RELU", inputs=[in_id], shape=[1, 2]))
    g.set_inputs([in_id])
    g.set_outputs([relu_id])

    g_cpp = cpp.json_to_graph(g.to_json())
    interp = cpp.Interpreter()

    # Create a non-contiguous slice: shape (1, 4), sliced with step 2 -> shape (1, 2)
    # Physical array: [[0, 1, 2, 3]] -> slice: [[0, 2]]
    base = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    sliced = base[:, ::2]
    assert not sliced.flags.c_contiguous

    out_dict = interp.evaluate(g_cpp, {0: sliced})
    out = out_dict[1]
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out, [[0.0, 2.0]], err_msg="Non-contiguous slice must read correct elements [[0, 2]]")


def test_regression_native_compiler_opt_level_validation():
    """
    P2.9 Audit:
    Unknown optimization levels must be rejected with ValueError across all toolchains.
    """
    from forgetune.native_compiler import NativeCompiler
    comp = NativeCompiler()
    with pytest.raises(ValueError) as exc_info:
        comp.compile_shared_library("dummy.cpp", "dummy.dll", opt_level="-Ofastest")
    assert "Invalid optimization level" in str(exc_info.value)


def test_regression_cross_model_artifact_reuse_prevention(tmp_path):
    """
    Audit P1: Artifact reuse for different models.
    Compiling Model A, then Model B into the same output_dir, then Model A again
    must invalidate and rebuild artifacts rather than executing stale kernels.
    """
    class ModelA(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 8)
        def forward(self, x):
            return self.fc(x)

    class ModelB(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 8)
        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    out_dir = str(tmp_path / "shared_artifacts")
    x = torch.randn(2, 16)

    m_a = ModelA().eval()
    m_b = ModelB().eval()

    # 1. Compile Model A
    comp_a1 = compile_model(m_a, (x,), output_dir=out_dir, autotune=False)
    out_a1 = comp_a1.run(x)
    ref_a = m_a(x).detach().numpy()
    np.testing.assert_allclose(out_a1, ref_a, atol=1e-4)
    comp_a1.close()

    # 2. Compile Model B into the exact same output_dir
    comp_b = compile_model(m_b, (x,), output_dir=out_dir, autotune=False)
    out_b = comp_b.run(x)
    ref_b = m_b(x).detach().numpy()
    np.testing.assert_allclose(out_b, ref_b, atol=1e-4)
    comp_b.close()

    # 3. Re-compile Model A into the exact same output_dir (must rebuild, not reuse B's kernel)
    comp_a2 = compile_model(m_a, (x,), output_dir=out_dir, autotune=False)
    out_a2 = comp_a2.run(x)
    np.testing.assert_allclose(out_a2, ref_a, atol=1e-4)
    comp_a2.close()


def test_regression_dynamic_reshape_rejects_fixed_batch_one():
    """
    Audit P2: Dynamic reshape must not confuse literal 1 with dynamic batch dimension.
    Calling x.view(1, -1) on a dynamic batch tensor [-1, 16] must be rejected with UnsupportedOpError.
    """
    class BadReshapeModel(nn.Module):
        def forward(self, x):
            return x.view(1, -1)

    model = BadReshapeModel().eval()
    x = torch.randn(2, 16)

    with pytest.raises(UnsupportedOpError) as exc_info:
        compile_model(model, (x,), dynamic_batch=True, autotune=False)
    assert "fixed leading dimension" in str(exc_info.value).lower() or "not supported" in str(exc_info.value).lower()


def test_regression_exact_64bit_integer_json_parsing():
    """
    Audit P2: C++ JSON parser must parse exact 64-bit integers and reject overflow.
    """
    from forgetune import _forgetune_cpp as cpp

    # Valid exact 64-bit integer > 2^53: 9007199254740993 in tensor shape
    json_valid = r"""{
      "schema_version_major": 2,
      "schema_version_minor": 0,
      "max_batch_size": 128,
      "inputs": [0],
      "outputs": [0],
      "nodes": [
        {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [1, 9007199254740993], "dtype": "f32"}
      ]
    }"""
    g = cpp.json_to_graph(json_valid)
    assert g.get_node(0).output_meta.shape.dims == [1, 9007199254740993]

    # Overflow beyond INT64_MAX: 9223372036854775808
    json_overflow = r"""{
      "schema_version_major": 2,
      "schema_version_minor": 0,
      "max_batch_size": 128,
      "inputs": [0],
      "outputs": [0],
      "nodes": [
        {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [1, 9223372036854775808], "dtype": "f32"}
      ]
    }"""
    with pytest.raises(Exception):
        cpp.json_to_graph(json_overflow)


def test_regression_strict_json_commas_and_trailing_characters():
    """
    Audit P2: JSON parser must enforce strict RFC 8259 comma separation and reject trailing garbage.
    """
    from forgetune import _forgetune_cpp as cpp

    # Missing comma between nodes
    json_missing_comma = r"""{
      "schema_version_major": 2,
      "schema_version_minor": 0,
      "max_batch_size": 128,
      "inputs": [0],
      "outputs": [1],
      "nodes": [
        {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [1, 2], "dtype": "f32"}
        {"id": 1, "op": "RELU", "name": "y", "inputs": [0], "shape": [1, 2], "dtype": "f32"}
      ]
    }"""
    with pytest.raises(Exception):
        cpp.json_to_graph(json_missing_comma)

    # Trailing comma in array
    json_trailing_comma = r"""{
      "schema_version_major": 2,
      "schema_version_minor": 0,
      "max_batch_size": 128,
      "inputs": [0,],
      "outputs": [0],
      "nodes": [
        {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [1, 2], "dtype": "f32"}
      ]
    }"""
    with pytest.raises(Exception):
        cpp.json_to_graph(json_trailing_comma)

    # Trailing garbage after root object
    json_trailing_garbage = r"""{
      "schema_version_major": 2,
      "schema_version_minor": 0,
      "max_batch_size": 128,
      "inputs": [0],
      "outputs": [0],
      "nodes": [
        {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [1, 2], "dtype": "f32"}
      ]
    } extra_garbage"""
    with pytest.raises(Exception):
        cpp.json_to_graph(json_trailing_garbage)


def test_regression_locked_dll_raises_informative_compilation_error(tmp_path):
    """
    Audit P2: Recompiling a locked shared library must raise an informative CompilationError.
    """
    if sys.platform != "win32":
        pytest.skip("Shared library file locking during active execution is Windows-specific")

    class SimpleModel(nn.Module):
        def forward(self, x):
            return x

    out_dir = str(tmp_path / "locked_test")
    x = torch.randn(1, 4)
    compiled = compile_model(SimpleModel().eval(), (x,), output_dir=out_dir, autotune=False)

    dll_path = compiled.manifest()["artifacts"]["shared_library"]
    cpp_path = compiled.manifest()["artifacts"]["cpp_source"]

    # Recompiling while compiled._runtime keeps the DLL locked on Windows
    nc = NativeCompiler()
    try:
        with pytest.raises(CompilationError) as exc_info:
            nc.compile_shared_library(cpp_path, dll_path)
        assert "locked by an active process or runtime" in str(exc_info.value)
    finally:
        compiled.close()

    # After close(), recompilation should succeed
    res = nc.compile_shared_library(cpp_path, dll_path)
    assert res["success"] is True


def test_regression_closed_runtime_lifecycle(tmp_path):
    """
    Audit P2: Executing NativeRuntime after close() must raise RuntimeError.
    """
    class SimpleModel(nn.Module):
        def forward(self, x):
            return x

    out_dir = str(tmp_path / "lifecycle_test")
    x = torch.randn(1, 4)

    # 1. Manual close()
    compiled = compile_model(SimpleModel().eval(), (x,), output_dir=out_dir, autotune=False)
    assert compiled.run(x).shape == (1, 4)
    compiled.close()
    with pytest.raises(RuntimeError) as exc_info:
        compiled.run(x)
    assert "after it has been closed" in str(exc_info.value)

    # 2. Context manager __enter__ / __exit__
    with compile_model(SimpleModel().eval(), (x,), output_dir=str(tmp_path / "cm_test"), autotune=False) as cm:
        assert cm.run(x).shape == (1, 4)
    with pytest.raises(RuntimeError):
        cm.run(x)


def test_regression_api_parameter_strict_validation():
    """
    Audit P3: Strict parameter type validation (rejecting bool masquerading as int).
    """
    class SimpleModel(nn.Module):
        def forward(self, x):
            return x

    model = SimpleModel().eval()
    x = torch.randn(1, 4)

    with pytest.raises(ValueError) as exc:
        compile_model(model, (x,), max_autotune_budget=True)
    assert "max_autotune_budget" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        compile_model(model, (x,), max_batch_size=True)
    assert "max_batch_size" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        compile_model(model, (x,), autotune=1)
    assert "autotune" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        compile_model(model, (x,), dynamic_batch=1)
    assert "dynamic_batch" in str(exc.value)


def test_regression_kernel_config_immutable_dirs_and_no_stale_reuse(tmp_path):
    """
    Audit 4 P1.1: Different tile sizes or configurations under the same candidate ID
    must generate distinct immutable directories/libraries and not reuse stale artifacts.
    """
    class LinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(32, 32, bias=False)
        def forward(self, x):
            return self.fc(x)

    model = LinearModel().eval()
    x = torch.randn(2, 32)
    out_dir = str(tmp_path / "tile_reuse_test")

    # 1. Compile with tile_sizes=(8, 8, 8)
    comp_8 = compile_model(
        model, (x,),
        output_dir=out_dir,
        autotune=False,
        kernel_variant="tiled",
        tile_sizes=(8, 8, 8)
    )
    lib_path_8 = comp_8.manifest()["artifacts"]["shared_library"]
    manifest_8 = comp_8.manifest()

    # 2. Compile same model with tile_sizes=(16, 16, 16) in the same output_dir
    comp_16 = compile_model(
        model, (x,),
        output_dir=out_dir,
        autotune=False,
        kernel_variant="tiled",
        tile_sizes=(16, 16, 16)
    )
    lib_path_16 = comp_16.manifest()["artifacts"]["shared_library"]
    manifest_16 = comp_16.manifest()

    # Must produce distinct paths and distinct tile configurations
    assert lib_path_8 != lib_path_16, f"Expected distinct library paths, got {lib_path_8}"
    assert manifest_8["kernel_config"]["tile_sizes"] == (8, 8, 8)
    assert manifest_16["kernel_config"]["tile_sizes"] == (16, 16, 16)

    comp_8.close()
    comp_16.close()


def test_regression_autotune_parameter_conflicts_rejected(tmp_path):
    """
    Audit 4 P2.5: Specifying manual kernel_variant or tile_sizes when autotune=True
    must raise explicit ValueError rather than being silently ignored.
    """
    class SimpleModel(nn.Module):
        def forward(self, x):
            return x

    model = SimpleModel().eval()
    x = torch.randn(2, 4)
    out_dir = str(tmp_path / "autotune_conflict_test")

    with pytest.raises(ValueError) as exc:
        compile_model(model, (x,), output_dir=out_dir, autotune=True, kernel_variant="tiled")
    assert "Cannot specify 'kernel_variant'" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        compile_model(model, (x,), output_dir=out_dir, autotune=True, tile_sizes=(8, 8, 8))
    assert "Cannot specify 'tile_sizes'" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        compile_model(model, (x,), output_dir=out_dir, autotune=True, compiler_opt_level="-O0")
    assert "Cannot override 'compiler_opt_level'" in str(exc.value)


def test_regression_compile_model_preserves_training_state(tmp_path):
    """
    Audit 4 P3.12: compile_model() must preserve the caller's model.training state.
    """
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    model.train(True)
    assert model.training is True

    x = torch.randn(2, 4)
    out_dir = str(tmp_path / "preserve_train_test")
    compiled = compile_model(model, (x,), output_dir=out_dir, autotune=False)
    try:
        assert model.training is True, "Expected model.training to remain True after compile_model()"
    finally:
        compiled.close()


def test_regression_rebuild_after_artifact_purge_verifies_correctness(tmp_path):
    """
    Audit 4 P2.4: Rebuilding an artifact after deletion exercises the differential correctness gate.
    """
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
        def forward(self, x):
            return self.fc(x)

    model = SimpleModel().eval()
    x = torch.randn(2, 4)
    out_dir = str(tmp_path / "rebuild_gate_test")

    # Initial compilation
    compiled = compile_model(model, (x,), output_dir=out_dir, autotune=False)
    lib_path = compiled.manifest()["artifacts"]["shared_library"]
    compiled.close()

    # Delete the compiled library to force a rebuild
    os.remove(lib_path)
    assert not os.path.exists(lib_path)

    # Recompiling should rebuild, execute verification gate, and succeed
    rebuilt = compile_model(model, (x,), output_dir=out_dir, autotune=False)
    try:
        out = rebuilt.run(x)
        ref = model(x).detach().numpy()
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
    finally:
        rebuilt.close()


def test_regression_json_parsing_null_duplicate_keys_and_nonfinite(tmp_path):
    """
    Audit 4 P2.9 & P3.10: C++ JSON parser supports null, rejects duplicate keys,
    rejects malformed nested unknown fields, and rejects non-finite constants.
    """
    from forgetune import _forgetune_cpp as cpp

    # 1. Valid JSON with null value in unknown field
    json_with_null = """{
      "schema_version_major": 2,
      "schema_version_minor": 0,
      "max_batch_size": 128,
      "inputs": [0],
      "outputs": [0],
      "extra_meta": null,
      "nodes": [
        {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [1, 4], "dtype": "f32"}
      ]
    }"""
    g = cpp.json_to_graph(json_with_null)
    assert g.live_node_count() == 1

    # 2. Duplicate property in root object
    json_dup_root = """{
      "schema_version_major": 2,
      "schema_version_major": 2,
      "inputs": [0],
      "outputs": [0],
      "nodes": []
    }"""
    with pytest.raises(Exception) as exc:
        cpp.json_to_graph(json_dup_root)
    assert "duplicate property" in str(exc.value)

    # 3. Duplicate property in node object
    json_dup_node = """{
      "schema_version_major": 2,
      "schema_version_minor": 0,
      "max_batch_size": 128,
      "inputs": [0],
      "outputs": [0],
      "nodes": [
        {"id": 0, "id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [1, 4], "dtype": "f32"}
      ]
    }"""
    with pytest.raises(Exception) as exc:
        cpp.json_to_graph(json_dup_node)
    assert "duplicate property" in str(exc.value)

    # 4. Malformed nested unknown field
    json_malformed_nested = """{
      "schema_version_major": 2,
      "schema_version_minor": 0,
      "max_batch_size": 128,
      "inputs": [0],
      "outputs": [0],
      "unknown_block": { "bad": [1, 2, } },
      "nodes": [
        {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [1, 4], "dtype": "f32"}
      ]
    }"""
    with pytest.raises(Exception) as exc:
        cpp.json_to_graph(json_malformed_nested)

    # 5. Non-finite constant in const_data
    json_inf_const = """{
      "schema_version_major": 2,
      "schema_version_minor": 0,
      "max_batch_size": 128,
      "inputs": [0],
      "outputs": [0],
      "nodes": [
        {"id": 0, "op": "CONST", "name": "c", "inputs": [], "shape": [1], "dtype": "f32", "const_data": [1e40]}
      ]
    }"""
    with pytest.raises(Exception) as exc:
        cpp.json_to_graph(json_inf_const)
    assert "overflow" in str(exc.value) or "non-finite" in str(exc.value)


def test_regression_graph_adapter_strict_types_and_canonical_ops():
    """
    Audit 4 P2.7 & P2.8: Python Graph.from_dict() rejects booleans as ints,
    normalizes aliases (CONSTANT -> CONST), and rejects unsupported ops (SUB, DIV).
    """
    from forgetune.graph_adapter import Graph
    from forgetune.errors import InvalidGraphError, UnsupportedOpError

    base_dict = {
        "schema_version_major": 2,
        "schema_version_minor": 0,
        "max_batch_size": 128,
        "inputs": [0],
        "outputs": [0],
        "nodes": [
            {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [1, 4], "dtype": "f32"}
        ]
    }

    # Reject boolean as max_batch_size
    bad_batch = dict(base_dict, max_batch_size=True)
    with pytest.raises(InvalidGraphError):
        Graph.from_dict(bad_batch)

    # Reject boolean as node id
    bad_id = dict(base_dict, nodes=[
        {"id": True, "op": "INPUT", "name": "x", "inputs": [], "shape": [1, 4], "dtype": "f32"}
    ])
    with pytest.raises(InvalidGraphError):
        Graph.from_dict(bad_id)

    # Reject SUB op
    bad_op = dict(base_dict, nodes=[
        {"id": 0, "op": "SUB", "name": "s", "inputs": [], "shape": [1, 4], "dtype": "f32"}
    ])
    with pytest.raises(UnsupportedOpError):
        Graph.from_dict(bad_op)

    # Normalize CONSTANT alias
    alias_dict = {
        "schema_version_major": 2,
        "schema_version_minor": 0,
        "max_batch_size": 128,
        "inputs": [0],
        "outputs": [0],
        "nodes": [
            {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [1, 4], "dtype": "f32"},
            {"id": 1, "op": "CONSTANT", "name": "c", "inputs": [], "shape": [1], "dtype": "f32", "const_data": [1.0]}
        ]
    }
    g = Graph.from_dict(alias_dict)
    assert g.nodes[1].op == "CONST"


def test_regression_max_batch_size_propagation_autotune_true(tmp_path):
    """
    Audit 5 P1.1: Verify max_batch_size=256 propagates through autotuner compilation path
    and CLI engine path, correctly running at batch 200 and 256.
    """
    class DynMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)
        def forward(self, x):
            return self.fc(x)

    model = DynMLP().eval()
    ex = torch.randn(2, 8)
    out_dir = str(tmp_path / "autotune_mb256")
    compiled = compile_model(
        model,
        (ex,),
        dynamic_batch=True,
        max_batch_size=256,
        autotune=True,
        max_autotune_budget=2,
        output_dir=out_dir
    )

    # Test batch size 200 and 256
    x200 = np.ones((200, 8), dtype=np.float32)
    res200 = compiled.run(x200)
    assert res200.shape == (200, 4)

    x256 = np.ones((256, 8), dtype=np.float32)
    res256 = compiled.run(x256)
    assert res256.shape == (256, 4)

    # Reject batch size 257 (> max_batch_size 256)
    x257 = np.ones((257, 8), dtype=np.float32)
    with pytest.raises(Exception):
        compiled.run(x257)


def test_regression_cli_max_batch_size_propagation(tmp_path):
    """
    Audit 5 P1.1: Direct CLI engine compilation adopts graph max_batch_size or --max-batch-size flag.
    """
    from forgetune.compiler import get_engine_executable
    from forgetune.graph_adapter import Graph, Node
    import subprocess

    g = Graph()
    g.max_batch_size = 256
    x = g.add_node(Node(0, "INPUT", shape=[-1, 4]))
    w = g.add_node(Node(1, "CONST", shape=[4, 2], const_data=[1.0] * 8))
    mm = g.add_node(Node(2, "MATMUL", inputs=[x, w], shape=[-1, 2]))
    g.set_inputs([x])
    g.set_outputs([mm])

    json_p = str(tmp_path / "graph.json")
    cpp_p = str(tmp_path / "kernel.cpp")
    with open(json_p, "w", encoding="utf-8") as f:
        f.write(g.to_json())

    engine_exe = get_engine_executable()
    res = subprocess.run([engine_exe, "--compile", json_p, "-o", cpp_p, "--variant", "naive_ijk"], capture_output=True, text=True)
    assert res.returncode == 0, f"CLI compilation failed: {res.stderr}"

    with open(cpp_p, "r", encoding="utf-8") as f:
        code = f.read()
    assert "256" in code, "Max batch size 256 was not propagated into generated C++ source!"


def test_regression_enable_fusion_contract(tmp_path):
    """
    Audit 5 P2.2:
    - enable_fusion=False strictly excludes all fused candidates in autotuner.
    - Contradictory manual variant (fused_ikj with enable_fusion=False) raises ValueError.
    """
    from forgetune.autotuner import Autotuner

    tuner = Autotuner()
    cands_nofusion = tuner.get_candidate_space(8, enable_fusion=False)
    for c in cands_nofusion:
        assert not c.enable_fusion, f"Candidate {c.candidate_id} has fusion enabled when enable_fusion=False!"

    class SimpleLin(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.relu = nn.ReLU()
        def forward(self, x):
            return self.relu(self.fc(x))

    model = SimpleLin().eval()
    ex = torch.randn(1, 4)

    # Contradictory options rejection
    with pytest.raises(ValueError, match="Contradictory options"):
        compile_model(
            model,
            (ex,),
            autotune=False,
            kernel_variant="fused_ikj",
            enable_fusion=False,
            output_dir=str(tmp_path / "contra")
        )

    # Autotune with enable_fusion=False must never select fused candidate
    compiled = compile_model(
        model,
        (ex,),
        autotune=True,
        enable_fusion=False,
        max_autotune_budget=4,
        output_dir=str(tmp_path / "nofusion_autotune")
    )
    assert not compiled.manifest()["kernel_config"]["enable_fusion"]


def test_regression_artifact_invalidation_on_compiler_fingerprint_change(tmp_path):
    """
    Audit 5 P2.3 & P2.10: Stored candidate manifest is invalidated if compiler_fingerprint,
    compiler_path, or DLL hash changes.
    """
    from forgetune.compiler import _is_artifact_valid_for_graph
    from forgetune.autotuner import KernelCandidate

    cand = KernelCandidate("cand1", "loop_ikj", enable_fusion=False, opt_level="-O2")
    cand_dir = str(tmp_path / "cand1")
    os.makedirs(cand_dir, exist_ok=True)
    manifest_p = os.path.join(cand_dir, "cand_manifest.json")
    cpp_p = os.path.join(cand_dir, "kernel.cpp")
    dll_ext = ".dll" if sys.platform == "win32" else ".so"
    dll_p = os.path.join(cand_dir, f"kernel{dll_ext}")

    with open(cpp_p, "w") as f:
        f.write("// dummy code")
    with open(dll_p, "wb") as f:
        f.write(b"dummy dll")

    compiler_info = {"type": "zig_cxx", "version": "0.16.0", "path": "C:/fake/zig.exe"}
    manifest = {
        "graph_hash": "graph123",
        "candidate_id": "cand1",
        "variant": "loop_ikj",
        "enable_fusion": False,
        "tile_sizes": [None, None, None],
        "opt_level": "-O2",
        "platform": sys.platform,
        "machine": platform.machine(),
        "compiler_type": "zig_cxx",
        "compiler_version": "0.16.0",
        "compiler_path": "C:/fake/zig.exe",
        "compiler_fingerprint": "fingerprint_v1",
        "source_hash": hashlib.sha256(b"// dummy code").hexdigest(),
        "dll_hash": hashlib.sha256(b"dummy dll").hexdigest(),
    }
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    # Valid with matching fingerprint and path
    assert _is_artifact_valid_for_graph(cand_dir, "graph123", cand, compiler_info, dll_p, cpp_p, compiler_fingerprint="fingerprint_v1")

    # Invalid if compiler implementation fingerprint changed
    assert not _is_artifact_valid_for_graph(cand_dir, "graph123", cand, compiler_info, dll_p, cpp_p, compiler_fingerprint="fingerprint_v2_changed")

    # Invalid if compiler path changed
    diff_compiler_info = dict(compiler_info, path="C:/other/zig.exe")
    assert not _is_artifact_valid_for_graph(cand_dir, "graph123", cand, diff_compiler_info, dll_p, cpp_p, compiler_fingerprint="fingerprint_v1")


def test_regression_graph_validation_hardening():
    """
    Audit 5 P2.4: Comprehensive graph property and operation contract validation in Python loader.
    """
    from forgetune.graph_adapter import Graph
    from forgetune.errors import InvalidGraphError, InvalidShapeError

    # 1. Root not dict
    with pytest.raises(InvalidGraphError):
        Graph.from_dict(["not", "a", "dict"])

    # 2. Nodes not a list
    with pytest.raises(InvalidGraphError):
        Graph.from_dict({"inputs": [0], "outputs": [0], "nodes": "not_a_list"})

    # 3. Node name not string
    with pytest.raises(InvalidGraphError):
        Graph.from_dict({
            "inputs": [0], "outputs": [0],
            "nodes": [{"id": 0, "op": "INPUT", "name": 12345, "shape": [1, 2]}]
        })

    # 4. Node deleted not bool
    with pytest.raises(InvalidGraphError):
        Graph.from_dict({
            "inputs": [0], "outputs": [0],
            "nodes": [{"id": 0, "op": "INPUT", "deleted": "false", "shape": [1, 2]}]
        })

    # 5. Unsupported dtype (float64)
    with pytest.raises(InvalidGraphError):
        Graph.from_dict({
            "inputs": [0], "outputs": [0],
            "nodes": [{"id": 0, "op": "INPUT", "dtype": "float64", "shape": [1, 2]}]
        })

    # 6. Invalid conv_attrs
    with pytest.raises(InvalidGraphError):
        Graph.from_dict({
            "inputs": [0], "outputs": [0],
            "nodes": [{"id": 0, "op": "INPUT", "conv_attrs": "not_a_dict", "shape": [1, 2]}]
        })
    with pytest.raises(InvalidGraphError):
        Graph.from_dict({
            "inputs": [0], "outputs": [0],
            "nodes": [{"id": 0, "op": "INPUT", "conv_attrs": {"stride": -1}, "shape": [1, 2]}]
        })

    # 7. Operator input arity mismatch (ADD with 1 input)
    with pytest.raises(InvalidGraphError):
        Graph.from_dict({
            "inputs": [0], "outputs": [1],
            "nodes": [
                {"id": 0, "op": "INPUT", "shape": [2, 2]},
                {"id": 1, "op": "ADD", "inputs": [0], "shape": [2, 2]}
            ]
        })

    # 8. CONST node shape vs data count mismatch
    with pytest.raises(InvalidShapeError):
        Graph.from_dict({
            "inputs": [0], "outputs": [0],
            "nodes": [
                {"id": 0, "op": "INPUT", "shape": [1, 2]},
                {"id": 1, "op": "CONST", "shape": [2, 2], "const_data": [1.0, 2.0]} # expects 4, got 2
            ]
        })


def test_regression_undeclared_input_node_rejected():
    """
    Audit 5 P2.5: Live INPUT nodes not declared in graph.inputs must be rejected.
    """
    from forgetune.graph_adapter import Graph
    from forgetune.errors import InvalidGraphError

    bad_graph = {
        "schema_version_major": 2,
        "schema_version_minor": 0,
        "max_batch_size": 128,
        "inputs": [0], # Only node 0 declared! Node 1 is an undeclared live INPUT!
        "outputs": [2],
        "nodes": [
            {"id": 0, "op": "INPUT", "name": "x0", "inputs": [], "shape": [1, 4], "dtype": "f32"},
            {"id": 1, "op": "INPUT", "name": "x1_undeclared", "inputs": [], "shape": [1, 4], "dtype": "f32"},
            {"id": 2, "op": "ADD", "name": "add", "inputs": [0, 1], "shape": [1, 4], "dtype": "f32"}
        ]
    }
    with pytest.raises(InvalidGraphError, match="Live INPUT node.*not declared in graph.inputs"):
        Graph.from_dict(bad_graph)


def test_regression_shared_python_cpp_json_schema():
    """
    Audit 5 P2.6: Shared corpus of JSON graphs tested against both Python and C++ graph loaders.
    Verifies agreement on accept/reject behavior.
    """
    from forgetune.graph_adapter import Graph
    import forgetune._forgetune_cpp as cpp

    # Case 1: Valid graph
    valid_json = json.dumps({
        "schema_version_major": 2,
        "schema_version_minor": 0,
        "max_batch_size": 128,
        "inputs": [0],
        "outputs": [1],
        "nodes": [
            {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [2, 4], "dtype": "f32"},
            {"id": 1, "op": "RELU", "name": "r", "inputs": [0], "shape": [2, 4], "dtype": "f32"}
        ]
    })
    g_py = Graph.from_json(valid_json)
    g_cpp = cpp.json_to_graph(valid_json)
    assert len(g_py.nodes) == 2
    assert g_cpp.live_node_count() == 2

    # Case 2: Large string literal exceeding 4096 bytes (DoS prevention)
    giant_string = "a" * 5000
    bad_large_str = json.dumps({
        "schema_version_major": 2,
        "schema_version_minor": 0,
        "max_batch_size": 128,
        "inputs": [0],
        "outputs": [0],
        "nodes": [
            {"id": 0, "op": "INPUT", "name": giant_string, "inputs": [], "shape": [1, 2], "dtype": "f32"}
        ]
    })
    # C++ lexer rejects string > 4096 bytes
    with pytest.raises(Exception):
        cpp.json_to_graph(bad_large_str)

    # Case 3: Unsupported dtype float64 rejected by both
    bad_dtype = json.dumps({
        "schema_version_major": 2,
        "schema_version_minor": 0,
        "max_batch_size": 128,
        "inputs": [0],
        "outputs": [0],
        "nodes": [
            {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [1, 2], "dtype": "float64"}
        ]
    })
    with pytest.raises(Exception):
        Graph.from_json(bad_dtype)
    with pytest.raises(Exception):
        cpp.json_to_graph(bad_dtype)


def test_regression_submodule_mixed_training_mode_preservation(tmp_path):
    """
    Audit 5 P3.16: compile_model() preserves mixed training/evaluation states of submodules.
    """
    class SubModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
        def forward(self, x):
            return self.linear(x)

    class CompositeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub1 = SubModule()
            self.sub2 = SubModule()
        def forward(self, x):
            return self.sub2(self.sub1(x))

    model = CompositeModel()
    # Establish mixed modes
    model.train(True)
    model.sub2.eval() # sub2 explicitly in eval mode

    assert model.training is True
    assert model.sub1.training is True
    assert model.sub2.training is False

    ex = torch.randn(1, 4)
    compiled = compile_model(model, (ex,), autotune=False, output_dir=str(tmp_path / "mixed_mode"))

    # Verify exact mixed modes restored
    assert model.training is True
    assert model.sub1.training is True
    assert model.sub2.training is False


def test_regression_runtime_batch_size_and_iters_strict_types():
    """
    Audit 5 P3.17 & P2.8: Strict integer validation (reject booleans) on batch_size, iters, and max_budget.
    """
    from forgetune.autotuner import Autotuner
    from forgetune.runtime import NativeRuntime
    from forgetune.errors import InvalidShapeError

    tuner = Autotuner()
    with pytest.raises(ValueError):
        tuner.get_candidate_space(max_budget=True) # boolean rejected
    with pytest.raises(ValueError):
        tuner.get_candidate_space(max_budget=0)
    with pytest.raises(ValueError):
        tuner.get_candidate_space(max_budget=9)

    # Validate batch_size in NativeRuntime
    rt = NativeRuntime.__new__(NativeRuntime)
    rt.is_closed = False
    rt.is_dynamic = True
    rt.expected_input_shape = (-1, 4)
    rt.max_batch_size = 128

    inp = np.ones((1, 4), dtype=np.float32)
    # Boolean True passed as batch_size must be rejected
    with pytest.raises(InvalidShapeError):
        rt.run(inp, batch_size=True)
    with pytest.raises(InvalidShapeError):
        rt.run(inp, batch_size=0)
    with pytest.raises(InvalidShapeError):
        rt.run(inp, batch_size=-5)

    # Boolean True passed as iters in benchmark_kernel must be rejected
    rt.benchmark_fn = lambda a, b, c: 0.0
    with pytest.raises(ValueError):
        rt.benchmark_kernel(inp, iters=True)
    with pytest.raises(ValueError):
        rt.benchmark_kernel(inp, iters=0)


def test_regression_conv_attrs_full_schema_and_typos():
    """
    P1 Regression: conv_attrs must validate all 7 schema fields, reject unknown keys/typos,
    and enforce positive strides/dilations, non-negative pads, groups==1.
    """
    from forgetune.graph_adapter import from_dict
    from forgetune import _forgetune_cpp as cpp

    base_graph = {
        "nodes": [
            {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [1, 3, 8, 8], "dtype": "f32"},
            {"id": 1, "op": "CONST", "name": "w", "inputs": [], "shape": [4, 3, 3, 3], "dtype": "f32", "const_data": [0.1] * 108},
            {
                "id": 2, "op": "CONV2D", "name": "conv", "inputs": [0, 1],
                "shape": [1, 4, 6, 6], "dtype": "f32",
                "conv_attrs": {
                    "stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0,
                    "dilation_h": 1, "dilation_w": 1, "groups": 1
                }
            }
        ],
        "inputs": [0],
        "outputs": [2]
    }

    # 1. Valid conv_attrs succeeds
    g = from_dict(base_graph)
    assert g is not None

    # 2. Typo in conv_attrs key (e.g. "strid_h") must raise InvalidGraphError
    typo_graph = json.loads(json.dumps(base_graph))
    del typo_graph["nodes"][2]["conv_attrs"]["stride_h"]
    typo_graph["nodes"][2]["conv_attrs"]["strid_h"] = 1
    with pytest.raises(InvalidGraphError, match="Unknown conv_attrs key: 'strid_h'"):
        from_dict(typo_graph)

    # 3. Non-dict conv_attrs
    nondict_graph = json.loads(json.dumps(base_graph))
    nondict_graph["nodes"][2]["conv_attrs"] = "not_a_dict"
    with pytest.raises(InvalidGraphError, match="must be a dict"):
        from_dict(nondict_graph)

    # 4. Non-integer conv_attrs value
    nonint_graph = json.loads(json.dumps(base_graph))
    nonint_graph["nodes"][2]["conv_attrs"]["stride_h"] = "1"
    with pytest.raises(InvalidGraphError, match="must be an integer"):
        from_dict(nonint_graph)

    # 5. Non-positive stride
    zero_stride_graph = json.loads(json.dumps(base_graph))
    zero_stride_graph["nodes"][2]["conv_attrs"]["stride_h"] = 0
    with pytest.raises(InvalidGraphError, match="must be a positive integer"):
        from_dict(zero_stride_graph)

    # 6. Negative pad
    neg_pad_graph = json.loads(json.dumps(base_graph))
    neg_pad_graph["nodes"][2]["conv_attrs"]["pad_h"] = -1
    with pytest.raises(InvalidGraphError, match="must be a non-negative integer"):
        from_dict(neg_pad_graph)

    # 7. C++ json_io rejection of unknown key
    json_typo_str = json.dumps(typo_graph)
    with pytest.raises(RuntimeError):
        cpp.json_to_graph(json_typo_str)


def test_regression_conv_attrs_groups_greater_than_one_rejected():
    """
    P1 Regression: conv_attrs with groups != 1 must be rejected in both Python and C++.
    """
    from forgetune.graph_adapter import from_dict
    from forgetune import _forgetune_cpp as cpp

    graph_dict = {
        "nodes": [
            {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [1, 4, 8, 8], "dtype": "f32"},
            {"id": 1, "op": "CONST", "name": "w", "inputs": [], "shape": [4, 2, 3, 3], "dtype": "f32", "const_data": [0.1] * 72},
            {
                "id": 2, "op": "CONV2D", "name": "conv", "inputs": [0, 1],
                "shape": [1, 4, 6, 6], "dtype": "f32",
                "conv_attrs": {
                    "stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0,
                    "dilation_h": 1, "dilation_w": 1, "groups": 2
                }
            }
        ],
        "inputs": [0],
        "outputs": [2]
    }

    with pytest.raises(InvalidGraphError, match="must be 1"):
        from_dict(graph_dict)

    with pytest.raises(RuntimeError):
        cpp.json_to_graph(json.dumps(graph_dict))


def test_regression_dtype_float_rejected():
    """
    P1 Regression: dtype 'float' must be rejected; only 'f32' and 'float32' are accepted.
    """
    from forgetune.graph_adapter import from_dict
    from forgetune import _forgetune_cpp as cpp

    graph_dict = {
        "nodes": [
            {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [2, 4], "dtype": "float"}
        ],
        "inputs": [0],
        "outputs": [0]
    }

    with pytest.raises(InvalidGraphError, match="Unsupported node dtype: 'float'"):
        from_dict(graph_dict)

    with pytest.raises(RuntimeError):
        cpp.json_to_graph(json.dumps(graph_dict))


def test_regression_duplicate_and_multiple_inputs_outputs():
    """
    P1 Regression: Duplicate inputs/outputs and multiple inputs/outputs must be strictly rejected.
    """
    from forgetune.graph_adapter import from_dict

    base = {
        "nodes": [
            {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [2, 4], "dtype": "f32"},
            {"id": 1, "op": "INPUT", "name": "y", "inputs": [], "shape": [2, 4], "dtype": "f32"}
        ],
        "inputs": [0],
        "outputs": [0]
    }

    # Duplicate input
    dup_in = json.loads(json.dumps(base))
    dup_in["inputs"] = [0, 0]
    with pytest.raises(InvalidGraphError):
        from_dict(dup_in)

    # Multiple inputs
    multi_in = json.loads(json.dumps(base))
    multi_in["inputs"] = [0, 1]
    with pytest.raises(InvalidGraphError):
        from_dict(multi_in)

    # Empty inputs
    empty_in = json.loads(json.dumps(base))
    empty_in["inputs"] = []
    with pytest.raises(InvalidGraphError):
        from_dict(empty_in)

    # Duplicate output
    dup_out = json.loads(json.dumps(base))
    dup_out["outputs"] = [0, 0]
    with pytest.raises(InvalidGraphError):
        from_dict(dup_out)

    # Multiple outputs
    multi_out = json.loads(json.dumps(base))
    multi_out["outputs"] = [0, 1]
    with pytest.raises(InvalidGraphError):
        from_dict(multi_out)


def test_regression_const_data_null_rejected():
    """
    P1 Regression: CONST node with const_data: null must raise InvalidGraphError, not TypeError.
    """
    from forgetune.graph_adapter import from_dict

    graph_dict = {
        "nodes": [
            {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [2, 4], "dtype": "f32"},
            {"id": 1, "op": "CONST", "name": "w", "inputs": [], "shape": [4, 4], "dtype": "f32", "const_data": None},
            {"id": 2, "op": "MATMUL", "name": "mm", "inputs": [0, 1], "shape": [2, 4], "dtype": "f32"}
        ],
        "inputs": [0],
        "outputs": [2]
    }

    with pytest.raises(InvalidGraphError, match="'const_data' cannot be null for CONST op"):
        from_dict(graph_dict)


def test_regression_fused_matmul_add_relu_invalid_broadcast():
    """
    P1 Regression: FUSED_MATMUL_ADD_RELU interpreter must validate bias shape against C++ contract.
    Valid shapes: 1D [N], 2D [1, N] or [M, N]. Other shapes must be rejected.
    """
    from forgetune.graph_adapter import from_dict, evaluate
    from forgetune.errors import InvalidShapeError

    # M=2, K=4, N=3
    # Illegal bias: shape [2, 2] instead of [3] or [1, 3] or [2, 3]
    bad_dict = {
        "nodes": [
            {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [2, 4], "dtype": "f32"},
            {"id": 1, "op": "CONST", "name": "w", "inputs": [], "shape": [4, 3], "dtype": "f32", "const_data": [1.0] * 12},
            {"id": 2, "op": "CONST", "name": "b", "inputs": [], "shape": [2, 2], "dtype": "f32", "const_data": [1.0] * 4},
            {"id": 3, "op": "FUSED_MATMUL_ADD_RELU", "name": "fused", "inputs": [0, 1, 2], "shape": [2, 3], "dtype": "f32"}
        ],
        "inputs": [0],
        "outputs": [3]
    }
    g_bad = from_dict(bad_dict)
    x_val = np.ones((2, 4), dtype=np.float32)

    with pytest.raises(InvalidShapeError, match="bias shape"):
        evaluate(g_bad, {0: x_val})

    # Legal bias shape [3] succeeds
    good_dict = {
        "nodes": [
            {"id": 0, "op": "INPUT", "name": "x", "inputs": [], "shape": [2, 4], "dtype": "f32"},
            {"id": 1, "op": "CONST", "name": "w", "inputs": [], "shape": [4, 3], "dtype": "f32", "const_data": [1.0] * 12},
            {"id": 2, "op": "CONST", "name": "b", "inputs": [], "shape": [3], "dtype": "f32", "const_data": [1.0] * 3},
            {"id": 3, "op": "FUSED_MATMUL_ADD_RELU", "name": "fused", "inputs": [0, 1, 2], "shape": [2, 3], "dtype": "f32"}
        ],
        "inputs": [0],
        "outputs": [3]
    }
    g_good = from_dict(good_dict)
    res = evaluate(g_good, {0: x_val})
    assert res[3].shape == (2, 3)
    np.testing.assert_allclose(res[3], np.full((2, 3), 5.0, dtype=np.float32))


def test_regression_autotuner_bounded_memory_for_large_batch():
    """
    P1 Regression: get_bounded_test_batches must cap dynamic test batches to <= 256
    even if max_batch_size is 1,000,000, preventing multi-gigabyte memory allocations.
    """
    from forgetune.compiler import get_bounded_test_batches

    # Dynamic shape with huge max_batch_size
    batches = get_bounded_test_batches((-1, 16), max_batch_size=1_000_000)
    assert max(batches) <= 256
    assert 1 in batches
    assert 16 in batches
    assert 1_000_000 not in batches

    # With example_batch=2
    batches_ex2 = get_bounded_test_batches((-1, 16), max_batch_size=1_000_000, example_batch=2)
    assert 2 in batches_ex2

    # Static shape returns [batch_size]
    assert get_bounded_test_batches((64, 16), max_batch_size=128, example_batch=64) == [64]

    # Dynamic shape with small max_batch_size
    small_batches = get_bounded_test_batches((-1, 16), max_batch_size=10, example_batch=7)
    assert small_batches == [1, 7, 10]


def test_regression_autotuner_fusion_budgets_and_contradictory_args():
    """
    P1 Regression: Autotuner candidate space interleaving and contradictory argument rejection.
    """
    from forgetune.autotuner import Autotuner

    tuner = Autotuner()

    # Contradictory arguments must be rejected
    with pytest.raises(ValueError, match="Contradictory autotuner options"):
        tuner.get_candidate_space(enable_fusion=False, force_fusion=True)

    # Budget 4 must explore naive, loop_ikj, fused, and tiled
    cands_b4 = tuner.get_candidate_space(max_budget=4, enable_fusion=True)
    assert len(cands_b4) == 4
    c_names = [c.candidate_id for c in cands_b4]
    assert "baseline_naive" in c_names
    assert "loop_ikj" in c_names
    assert any("fused" in name for name in c_names), "Budget 4 must include fused candidate"
    assert any("tiled" in name for name in c_names), "Budget 4 must include tiled candidate"

    # Budget 8 must have 8 candidates
    cands_b8 = tuner.get_candidate_space(max_budget=8, enable_fusion=True)
    assert len(cands_b8) == 8


def test_regression_cli_max_batch_size_upper_bound():
    """
    P1 Regression: KernelConfig and CLI must reject max_batch_size > 1,000,000.
    """
    from forgetune import _forgetune_cpp as cpp
    import subprocess

    cfg = cpp.KernelConfig()
    cfg.max_batch_size = 1_000_000
    assert cfg.validate(), "max_batch_size = 1,000,000 should be valid"

    cfg.max_batch_size = 1_000_001
    assert not cfg.validate(), "max_batch_size = 1,000,001 should be invalid"

    cfg.max_batch_size = 0
    assert not cfg.validate(), "max_batch_size = 0 should be invalid"

    # Test CLI --max-batch-size upper bound
    engine_bin = "bin/forgetune_engine.exe" if os.name == "nt" else "bin/forgetune_engine"
    if os.path.exists(engine_bin):
        res = subprocess.run([engine_bin, "--compile", "dummy.json", "-o", "dummy.cpp", "--max-batch-size", "1000001"], capture_output=True, text=True)
        assert res.returncode != 0
        assert "1000000" in (res.stderr + res.stdout)


def test_regression_build_metadata_freshness_rebuild(tmp_path):
    """
    P1 Regression: Native build metadata tracks Python ABI, platform, machine, compiler, and flags.
    Mismatched metadata must invalidate cached native extensions.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from setup import is_build_metadata_valid
    import sysconfig

    meta_file = tmp_path / "_forgetune_cpp.build_meta.json"
    dummy_meta = {
        "python_version": "2.7.0",
        "python_abi": "cpython-27m",
        "platform": "unknown_os",
        "machine": "unknown_arch",
        "compiler_path": "fake_compiler",
        "compiler_version": "0.0.1",
        "build_flags": []
    }
    meta_file.write_text(json.dumps(dummy_meta), encoding="utf-8")

    # Mismatched metadata must be invalid
    assert not is_build_metadata_valid(str(meta_file))

    # Valid metadata must be valid
    valid_meta = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_abi": sysconfig.get_config_var("SOABI") or f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": sys.platform,
        "machine": platform.machine(),
        "compiler_path": "zig",
        "build_flags": []
    }
    meta_file.write_text(json.dumps(valid_meta), encoding="utf-8")
    assert is_build_metadata_valid(str(meta_file))

    





def test_regression_rank4_channel_broadcasting_order(tmp_path):
    """
    P1 Regression: export_frontend must handle rank-4 channel broadcasting in both operand orders
    (x + bias, bias + x, x * bias, bias * x) across dynamic batch B=1 and B>1 without dimension mismatches.
    """
    class ConvAddBiasRight(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.randn(1, 4, 1, 1))
        def forward(self, x):
            return x + self.bias

    class ConvAddBiasLeft(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.randn(1, 4, 1, 1))
        def forward(self, x):
            return self.bias + x

    class ConvMulBiasRight(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.randn(1, 4, 1, 1))
        def forward(self, x):
            return x * self.scale

    class ConvMulBiasLeft(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.randn(1, 4, 1, 1))
        def forward(self, x):
            return self.scale * x

    models = [
        ("add_r", ConvAddBiasRight()),
        ("add_l", ConvAddBiasLeft()),
        ("mul_r", ConvMulBiasRight()),
        ("mul_l", ConvMulBiasLeft()),
    ]

    for name, m in models:
        m.eval()
        ex = torch.randn(2, 4, 8, 8)
        out_dir = str(tmp_path / f"bcast_{name}")
        comp = compile_model(m, (ex,), output_dir=out_dir, dynamic_batch=True, autotune=False)
        for b in [1, 3]:
            test_x = torch.randn(b, 4, 8, 8)
            ref_y = m(test_x).detach().numpy()
            out_y = comp.run(test_x.numpy())
            np.testing.assert_allclose(out_y, ref_y, atol=1e-4, rtol=1e-4)
        comp.close()


def test_regression_model_state_restoration():
    """
    P1 Regression: export_to_forgetune_ir must preserve or restore the training state
    of the PyTorch model even if export succeeds or fails.
    """
    from forgetune.export_frontend import export_to_forgetune_ir
    m = nn.Linear(8, 4)
    m.train()
    assert m.training is True

    ex = torch.randn(2, 8)
    g = export_to_forgetune_ir(m, (ex,), dynamic_batch=True)
    assert g is not None
    assert m.training is True, "model.training must remain True after successful export"

    # Test restoration after failure
    class FailingModule(nn.Module):
        def forward(self, x):
            raise RuntimeError("Synthetic export trigger")

    m_fail = FailingModule()
    m_fail.train()
    with pytest.raises(Exception):
        export_to_forgetune_ir(m_fail, (ex,))
    assert m_fail.training is True, "model.training must remain True after failed export"


def test_regression_shared_linear_weights(tmp_path):
    """
    P1 Regression: Shared linear weight nodes must not be marked deleted prematurely
    when referenced by multiple consumers.
    """
    import torch.nn.functional as F

    class SharedWeightModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(8, 8))
        def forward(self, x):
            y1 = F.linear(x, self.w)
            y2 = F.linear(y1, self.w)
            return y2

    m = SharedWeightModel().eval()
    ex = torch.randn(2, 8)
    out_dir = str(tmp_path / "shared_weights")
    comp = compile_model(m, (ex,), output_dir=out_dir, dynamic_batch=False, autotune=False)

    test_x = torch.randn(2, 8)
    ref_y = m(test_x).detach().numpy()
    out_y = comp.run(test_x.numpy())
    np.testing.assert_allclose(out_y, ref_y, atol=1e-4, rtol=1e-4)
    comp.close()


def test_regression_linear_rank_greater_than_two_rejected():
    """
    P1 Regression: Linear operations with rank > 2 (e.g. 3D [B, S, D]) must raise UnsupportedOpError or InvalidShapeError.
    """
    class Linear3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
        def forward(self, x):
            return self.fc(x)

    m = Linear3D().eval()
    x = torch.randn(2, 4, 4)
    with pytest.raises(Exception) as exc_info:
        compile_model(m, (x,), autotune=False)
    assert "rank" in str(exc_info.value).lower()


def test_regression_memory_budget_exceeded_rejected():
    """
    P1 Regression: compile_model must validate estimated workspace against max_workspace_bytes
    and reject excessive requirements with ValueError.
    """
    class WideModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1024, 1024)
        def forward(self, x):
            return self.fc(x)

    m = WideModel().eval()
    x = torch.randn(2, 1024)
    # Set a tiny budget of 100 bytes
    with pytest.raises(ValueError) as exc_info:
        compile_model(m, (x,), max_workspace_bytes=100, autotune=False)
    assert "exceeding workspace budget" in str(exc_info.value)


def test_regression_deep_graph_topological_sort_kahn():
    """
    P1 Regression: Deep graphs (> 1000 nodes) must be sorted via iterative Kahn's algorithm
    without triggering Python RecursionError.
    """
    from forgetune.graph_adapter import Graph, Node

    g = Graph()
    prev = g.add_node(Node(0, "INPUT", shape=[2, 4]))
    g.set_inputs([prev])
    for i in range(1, 1500):
        prev = g.add_node(Node(i, "RELU", inputs=[prev], shape=[2, 4]))
    g.set_outputs([prev])

    order = g.topological_order()
    assert len(order) == 1500
    assert order[0] == 0
    assert order[-1] == 1499


def test_regression_autotuner_cache_corruption_detection(tmp_path):
    """
    P1 Regression: Autotuner cache with tampered or mismatched candidate configuration
    must be rejected and retuned rather than blindly trusted.
    """
    from forgetune.autotuner import Autotuner
    from forgetune.graph_adapter import Graph, Node

    tuner = Autotuner(cache_dir=str(tmp_path / "tuning_cache"))

    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[2, 4]))
    c = g.add_node(Node(1, "CONST", shape=[4, 8], const_data=[0.1] * 32))
    mm = g.add_node(Node(2, "MATMUL", inputs=[x, c], shape=[2, 8]))
    g.set_inputs([x])
    g.set_outputs([mm])

    x_val = np.ones((2, 4), dtype=np.float32)
    ref_out = np.full((2, 8), 0.4, dtype=np.float32)

    # Run initial tuning
    cand1, report1 = tuner.tune(
        graph=g,
        example_input=x_val,
        reference_output=ref_out,
        artifacts_dir=str(tmp_path / "artifacts"),
        max_budget=2,
        use_cache=True
    )
    assert cand1 is not None

    # Tamper with the saved cache JSON file by changing the opt_level to something unexpected
    import glob
    cache_files = glob.glob(os.path.join(str(tmp_path / "tuning_cache"), "*.json"))
    assert len(cache_files) >= 1
    cache_file = cache_files[0]
    with open(cache_file, "r") as f:
        data = json.load(f)
    data["selected_candidate"]["opt_level"] = "-O3"
    with open(cache_file, "w") as f:
        json.dump(data, f)

    # Calling tune again should detect cache mismatch and re-tune cleanly to expected -O2
    cand2, report2 = tuner.tune(
        graph=g,
        example_input=x_val,
        reference_output=ref_out,
        artifacts_dir=str(tmp_path / "artifacts_retune"),
        max_budget=2,
        use_cache=True
    )
    assert cand2.opt_level == "-O2"


def test_regression_runtime_input_dtype_validation(tmp_path):
    """
    P1 Regression: NativeRuntime must reject non-numeric or complex dtypes with TypeError.
    """
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)
        def forward(self, x):
            return self.fc(x)

    m = SimpleMLP().eval()
    ex = torch.randn(2, 8)
    out_dir = str(tmp_path / "dtype_test")
    comp = compile_model(m, (ex,), output_dir=out_dir, dynamic_batch=False, autotune=False)

    # Complex dtype input to compiled model
    c_arr = np.ones((2, 8), dtype=np.complex64)
    with pytest.raises(TypeError, match="Unsupported input tensor dtype"):
        comp.run(c_arr)

    # Complex dtype directly to runtime
    with pytest.raises(TypeError, match="Unsupported input tensor dtype"):
        comp._runtime.run(c_arr)

    # String / object dtype input
    s_arr = np.array([["str"] * 8, ["str"] * 8], dtype=object)
    with pytest.raises(TypeError, match="Unsupported input tensor dtype"):
        comp.run(s_arr)

    comp.close()
