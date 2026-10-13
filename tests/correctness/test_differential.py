"""
Category E: Differential Correctness Tests.
Compares:
  1. PyTorch model forward pass
  2. torch.export exported program execution
  3. ForgeTune Python reference interpreter
  4. Native unoptimized baseline kernel (naive_ijk)
  5. Native alternate loop ordering kernel (loop_ikj)
  6. Native tiled kernel (tiled_8, tiled_16)
  7. Native fused kernel (fused_ikj, fused_tiled)
  8. Final autotuner-selected implementation
"""

import os
import subprocess
import torch
import torch.nn as nn
import numpy as np
import pytest

from forgetune import compile_model, export_to_forgetune_ir
from forgetune.native_compiler import NativeCompiler
from forgetune.runtime import NativeRuntime

ENGINE_EXE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "forgetune_engine.exe"))

class DifferentialMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def test_full_differential_hierarchy():
    torch.manual_seed(42)
    np.random.seed(42)

    model = DifferentialMLP().eval()
    x_tensor = torch.randn(2, 4)
    x_np = x_tensor.numpy()

    # 1. PyTorch model reference
    with torch.no_grad():
        out_pytorch = model(x_tensor).numpy()

    # 2. Exported program reference
    exported = torch.export.export(model, (x_tensor,))
    with torch.no_grad():
        out_exported = exported.module()(x_tensor).numpy()
    assert np.allclose(out_pytorch, out_exported, atol=1e-6), "Exported program must match PyTorch"

    # 3. ForgeTune IR and reference interpreter
    ir_graph = export_to_forgetune_ir(model, (x_tensor,))
    out_interpreter = ir_graph.evaluate({ir_graph.inputs[0]: x_np})[ir_graph.outputs[0]]
    assert np.allclose(out_pytorch, out_interpreter, atol=1e-5), "Reference interpreter must match PyTorch"

    # Compile kernel variants directly
    os.makedirs("build/differential_test", exist_ok=True)
    json_path = "build/differential_test/graph.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(ir_graph.to_json())

    compiler = NativeCompiler()
    variants = [
        ("naive_ijk", False, ["--variant", "naive_ijk"]),
        ("loop_ikj", False, ["--variant", "loop_ikj"]),
        ("tiled_8", False, ["--variant", "tiled", "--tile-m", "8", "--tile-n", "8", "--tile-k", "8"]),
        ("tiled_16", False, ["--variant", "tiled", "--tile-m", "16", "--tile-n", "16", "--tile-k", "16"]),
        ("fused_ikj", True, ["--fusion", "--variant", "fused_ikj"]),
        ("fused_tiled", True, ["--fusion", "--variant", "fused_tiled", "--tile-m", "8", "--tile-n", "8", "--tile-k", "8"]),
    ]

    for name, fusion, extra_flags in variants:
        v_cpp = f"build/differential_test/{name}.cpp"
        v_dll = f"build/differential_test/{name}.dll"

        cmd = [ENGINE_EXE, "--compile", json_path, "-o", v_cpp] + extra_flags
        subprocess.run(cmd, check=True)

        compiler.compile_shared_library(v_cpp, v_dll)
        rt = NativeRuntime(v_dll, (2, 4), (2, 3))
        native_out = rt.run(x_np)

        max_abs = float(np.max(np.abs(native_out - out_pytorch)))
        assert max_abs < 1e-4, f"Variant '{name}' exceeded tolerance: max_abs={max_abs}"

    # 8. Autotuner selected compiled model
    compiled = compile_model(model, (x_tensor,), output_dir="build/differential_test/autotuned")
    autotuned_out = compiled.run(x_tensor)
    max_abs_auto = float(np.max(np.abs(autotuned_out - out_pytorch)))
    assert max_abs_auto < 1e-4, f"Autotuned model exceeded tolerance: max_abs={max_abs_auto}"
