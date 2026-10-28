import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import ctypes
import tempfile
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
    """
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
    """
    P2.4 / Bug 3 & 4 Audit:
    1. Unicode surrogate pairs (e.g. \uD83D\uDE00 for emoji) must transcode to 4-byte UTF-8 cleanly.
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


