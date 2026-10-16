"""
Category G: End-to-End Integration Tests for TinyMLP.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pytest
from forgetune import compile_model, read_manifest

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def test_tinymlp_full_lifecycle(tmp_path):
    torch.manual_seed(42)
    model = TinyMLP().eval()
    x = torch.randn(2, 4)

    output_dir = str(tmp_path / "e2e_tinymlp")
    compiled = compile_model(
        model=model,
        example_inputs=(x,),
        output_dir=output_dir,
        autotune=True
    )

    # 1. Output shape check
    out = compiled.run(x)
    assert out.shape == (2, 3), f"Expected shape (2, 3), got {out.shape}"

    # 2. Agreement with PyTorch reference
    with torch.no_grad():
        ref = model(x).numpy()
    max_diff = float(np.max(np.abs(out - ref)))
    assert max_diff < 1e-4, f"Mismatch with PyTorch reference: {max_diff}"

    # 3. Manifest validation
    manifest_path = os.path.join(output_dir, "manifest.json")
    assert os.path.isfile(manifest_path)
    manifest = read_manifest(manifest_path)
    assert manifest["model_name"] == "TinyMLP"
    assert manifest["input_shapes"] == [[2, 4]]
    assert manifest["output_shape"] == [2, 3]

    # 4. Benchmark execution
    bench = compiled.benchmark(warmup_iters=5, measure_iters=20)
    assert bench["median_latency_us"] > 0
    assert bench["measured_iterations"] == 20
