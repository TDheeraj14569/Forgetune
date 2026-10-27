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
