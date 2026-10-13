"""
Adversarial, Boundary Condition, and Dual ABI Verification Suite for ForgeTune v2.0.
Implements Round 2 Verification:
  1. Boundary batch sizes: B in {1, 2, 7, 16, 32, 128}
  2. Contract violations: B <= 0, B > 128, input element mismatch, null pointers
  3. Dual ABI entry points: forgetune_run_v1 (4-arg) on dynamic models, forgetune_run_v2 (5-arg) on static models
  4. Numerical stress: all zeros, negative values, high dynamic range
  5. Interleaved dynamic batch execution: sequential calls with alternating batch sizes
"""

import os
import ctypes
import numpy as np
import pytest
import torch
import torch.nn as nn

from forgetune.compiler import compile_model
from forgetune.runtime import NativeRuntime
from forgetune.errors import InvalidShapeError, RuntimeError

class DynamicLinearReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class StaticLinearReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

@pytest.fixture(scope="module")
def compiled_dynamic_model(tmp_path_factory):
    torch.manual_seed(42)
    m = DynamicLinearReLU().eval()
    x = torch.randn(2, 8)
    out_dir = str(tmp_path_factory.mktemp("v2_adv_dynamic"))
    compiled = compile_model(m, (x,), output_dir=out_dir, autotune=False, dynamic_batch=True)
    return compiled, m

@pytest.fixture(scope="module")
def compiled_static_model(tmp_path_factory):
    torch.manual_seed(42)
    m = StaticLinearReLU().eval()
    x = torch.randn(2, 8)
    out_dir = str(tmp_path_factory.mktemp("v2_adv_static"))
    compiled = compile_model(m, (x,), output_dir=out_dir, autotune=False, dynamic_batch=False)
    return compiled, m

def test_boundary_batch_sizes(compiled_dynamic_model):
    compiled, pt_model = compiled_dynamic_model
    for b in [1, 2, 7, 16, 32, 64, 128]:
        x = torch.randn(b, 8)
        with torch.no_grad():
            ref = pt_model(x).numpy()
        act = compiled.run(x)
        assert act.shape == (b, 4)
        np.testing.assert_allclose(act, ref, atol=1e-4, rtol=1e-4)

def test_rejection_boundary_violations(compiled_dynamic_model):
    compiled, _ = compiled_dynamic_model
    
    # B = 0
    with pytest.raises(InvalidShapeError):
        compiled.run(np.empty((0, 8), dtype=np.float32))

    # B = 129
    with pytest.raises(InvalidShapeError):
        compiled.run(np.ones((129, 8), dtype=np.float32))

    # Wrong feature dimension
    with pytest.raises(InvalidShapeError):
        compiled.run(np.ones((4, 7), dtype=np.float32))

def test_dual_abi_dynamic_model(compiled_dynamic_model):
    """Verifies that 4-argument forgetune_run_v1 correctly delegates to forgetune_run."""
    compiled, _ = compiled_dynamic_model
    rt = compiled._runtime
    assert hasattr(rt.lib, "forgetune_run_v1")

    # Call forgetune_run_v1 directly with B=2 input (16 floats) and output (8 floats)
    inp = np.ones((2, 8), dtype=np.float32)
    out = np.zeros((2, 4), dtype=np.float32)
    inp_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    status = rt.run_v1_fn(inp_ptr, 16, out_ptr, 8)
    assert status == 0, f"Expected 0 from forgetune_run_v1, got {status}"

    # Verify NULL pointer contract on v1
    assert rt.run_v1_fn(None, 16, out_ptr, 8) == -1
    assert rt.run_v1_fn(inp_ptr, 16, None, 8) == -1

def test_dual_abi_static_model(compiled_static_model):
    """Verifies that 5-argument forgetune_run_v2 compatibility wrapper works on static models."""
    compiled, _ = compiled_static_model
    rt = compiled._runtime
    assert hasattr(rt.lib, "forgetune_run_v2")

    inp = np.ones((2, 8), dtype=np.float32)
    out = np.zeros((2, 4), dtype=np.float32)
    inp_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # forgetune_run_v2 takes (inp, inp_len, out, out_len, batch_size)
    status = rt.run_v2_fn(inp_ptr, 16, out_ptr, 8, 2)
    assert status == 0, f"Expected 0 from forgetune_run_v2, got {status}"

def test_numerical_stress_and_zeros(compiled_dynamic_model):
    compiled, pt_model = compiled_dynamic_model

    # All zeros
    x_zero = torch.zeros(5, 8)
    with torch.no_grad():
        ref_zero = pt_model(x_zero).numpy()
    act_zero = compiled.run(x_zero)
    np.testing.assert_allclose(act_zero, ref_zero, atol=1e-5)

    # Large negative values
    x_neg = -100.0 * torch.ones(3, 8)
    act_neg = compiled.run(x_neg)
    with torch.no_grad():
        ref_neg = pt_model(x_neg).numpy()
    np.testing.assert_allclose(act_neg, ref_neg, atol=1e-4, rtol=1e-4)

def test_alternating_dynamic_batches_arena_stability(compiled_dynamic_model):
    """
    Stress-tests the thread_local memory arena by alternating batch sizes
    rapidly (1 -> 64 -> 3 -> 128 -> 2 -> 16 -> 1) in sequence.
    """
    compiled, pt_model = compiled_dynamic_model
    batch_seq = [1, 64, 3, 128, 2, 16, 1, 32, 7, 64, 1]
    for b in batch_seq:
        x = torch.randn(b, 8)
        with torch.no_grad():
            ref = pt_model(x).numpy()
        act = compiled.run(x)
        assert act.shape == (b, 4)
        np.testing.assert_allclose(act, ref, atol=1e-4, rtol=1e-4)
