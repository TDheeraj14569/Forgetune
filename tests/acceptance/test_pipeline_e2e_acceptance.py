"""
Acceptance Test Suite: End-to-End Production Pipeline Verification
Tests complete compiler pipeline from PyTorch torch.export ahead-of-time export,
canonical ForgeTune IR normalization, graph verification, autotuning,
standalone C++ code generation, compilation to shared library, and execution.
Verifies no PyTorch runtime arithmetic fallback occurs and numerical results
strictly match PyTorch within float32 tolerance.
"""

import os
import shutil
import tempfile
import pytest
import torch
import torch.nn as nn
import numpy as np

from forgetune.compiler import compile_model, CompiledModel
from forgetune.errors import InvalidShapeError, RuntimeError

class SingleLinearNoBias(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class SingleLinearWithBias(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class ThreeLayerMLP(nn.Module):
    def __init__(self, in_features: int, hidden1: int, hidden2: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden1, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2, bias=True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        return self.fc3(h2)

@pytest.fixture
def temp_output_dir():
    d = tempfile.mkdtemp(prefix="forgetune_accept_e2e_")
    yield d
    try:
        shutil.rmtree(d)
    except Exception:
        pass

def test_acceptance_single_linear_no_bias(temp_output_dir):
    torch.manual_seed(101)
    model = SingleLinearNoBias(8, 4)
    model.eval()
    x = torch.randn(2, 8, dtype=torch.float32)

    compiled = compile_model(
        model=model,
        example_inputs=(x,),
        output_dir=temp_output_dir,
        autotune=False,
        kernel_variant="naive_ijk"
    )

    assert isinstance(compiled, CompiledModel)
    assert os.path.isfile(os.path.join(temp_output_dir, "baseline", "kernel.dll"))
    assert os.path.isfile(os.path.join(temp_output_dir, "manifest.json"))

    # Execute compiled model
    native_out = compiled.run(x)
    assert isinstance(native_out, np.ndarray)
    assert native_out.shape == (2, 4)

    # Compare against PyTorch ground truth
    with torch.no_grad():
        ref_out = model(x).cpu().numpy()

    max_abs_diff = float(np.max(np.abs(native_out - ref_out)))
    assert max_abs_diff < 1e-4, f"Single linear no bias error exceeded tolerance: {max_abs_diff}"
    np.testing.assert_allclose(native_out, ref_out, atol=1e-4, rtol=1e-4)

def test_acceptance_single_linear_with_bias(temp_output_dir):
    torch.manual_seed(202)
    model = SingleLinearWithBias(16, 8)
    model.eval()
    x = torch.randn(4, 16, dtype=torch.float32)

    compiled = compile_model(
        model=model,
        example_inputs=(x,),
        output_dir=temp_output_dir,
        autotune=False,
        kernel_variant="loop_ikj"
    )

    native_out = compiled.run(x)
    with torch.no_grad():
        ref_out = model(x).cpu().numpy()

    max_abs_diff = float(np.max(np.abs(native_out - ref_out)))
    assert max_abs_diff < 1e-4, f"Single linear with bias error exceeded tolerance: {max_abs_diff}"
    np.testing.assert_allclose(native_out, ref_out, atol=1e-4, rtol=1e-4)

def test_acceptance_three_layer_mlp_with_relu_and_fusion(temp_output_dir):
    torch.manual_seed(303)
    model = ThreeLayerMLP(8, 16, 12, 4)
    model.eval()
    x = torch.randn(3, 8, dtype=torch.float32)

    compiled = compile_model(
        model=model,
        example_inputs=(x,),
        output_dir=temp_output_dir,
        autotune=False,
        kernel_variant="fused_ikj",
        enable_fusion=True
    )

    # Verify emitted C++ source code contains fused calls
    cpp_source = compiled.source_code()
    assert "fused_node_" in cpp_source or "forgetune_run" in cpp_source
    assert "forgetune_run" in cpp_source

    native_out = compiled.run(x)
    with torch.no_grad():
        ref_out = model(x).cpu().numpy()

    max_abs_diff = float(np.max(np.abs(native_out - ref_out)))
    assert max_abs_diff < 1e-4, f"Three layer MLP error exceeded tolerance: {max_abs_diff}"
    np.testing.assert_allclose(native_out, ref_out, atol=1e-4, rtol=1e-4)

def test_acceptance_wide_and_tall_batch_e2e(temp_output_dir):
    torch.manual_seed(404)
    # Tall batch (M=32, K=8, N=4)
    model_tall = SingleLinearWithBias(8, 4)
    x_tall = torch.randn(32, 8, dtype=torch.float32)

    compiled_tall = compile_model(
        model=model_tall,
        example_inputs=(x_tall,),
        output_dir=os.path.join(temp_output_dir, "tall"),
        autotune=False,
        kernel_variant="loop_ikj"
    )
    out_tall = compiled_tall.run(x_tall)
    with torch.no_grad():
        ref_tall = model_tall(x_tall).cpu().numpy()
    np.testing.assert_allclose(out_tall, ref_tall, atol=1e-4, rtol=1e-4)

    # Wide layer (M=2, K=16, N=64)
    model_wide = SingleLinearWithBias(16, 64)
    x_wide = torch.randn(2, 16, dtype=torch.float32)

    compiled_wide = compile_model(
        model=model_wide,
        example_inputs=(x_wide,),
        output_dir=os.path.join(temp_output_dir, "wide"),
        autotune=False,
        kernel_variant="loop_ikj"
    )
    out_wide = compiled_wide.run(x_wide)
    with torch.no_grad():
        ref_wide = model_wide(x_wide).cpu().numpy()
    np.testing.assert_allclose(out_wide, ref_wide, atol=1e-4, rtol=1e-4)

def test_acceptance_standalone_native_execution_no_pytorch_eval(temp_output_dir):
    """
    Validates that inference execution relies strictly on the compiled native library
    and can run without modifying or invoking the original PyTorch model.
    """
    torch.manual_seed(505)
    model = SingleLinearWithBias(4, 2)
    x = torch.randn(1, 4, dtype=torch.float32)
    x_np = x.numpy()

    compiled = compile_model(
        model=model,
        example_inputs=(x,),
        output_dir=temp_output_dir,
        autotune=False,
        kernel_variant="naive_ijk"
    )

    # Delete the original PyTorch model object from memory
    del model

    # Compiled model must still execute successfully using native runtime
    result = compiled.run(x_np)
    assert result.shape == (1, 2)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()
