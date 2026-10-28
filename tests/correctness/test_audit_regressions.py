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
    assert "invalid number format" in str(exc_info.value).lower()

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
