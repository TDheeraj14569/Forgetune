"""
ForgeTune v2.0 Example: Model A — Dynamic TinyMLP End-to-End Pipeline
Demonstrates:
  1. Defining Dynamic TinyMLP (16 -> 32 -> 8 with ReLU)
  2. Ahead-of-time export with dynamic batch dimension B in [1, 128]
  3. Compiling with ForgeTune self-optimizing engine
  4. Executing native C++ inference across dynamic batch sizes (1, 4, 16, 64, 128)
  5. Strict numerical validation against PyTorch eager reference
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
import forgetune

class DynamicTinyMLP(nn.Module):
    """
    Two-layer Multi-Layer Perceptron supporting dynamic leading batch:
      Layer 1: Linear(16, 32) + ReLU
      Layer 2: Linear(32, 8)
    """
    def __init__(self, in_features=16, hidden_features=32, out_features=8):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def main():
    print("=================================================================")
    print(" ForgeTune v2.0: Model A — Dynamic TinyMLP End-to-End")
    print("=================================================================")

    torch.manual_seed(42)
    model = DynamicTinyMLP().eval()

    example_input = torch.randn(2, 16)
    output_dir = os.path.join("build", "artifacts", "v2_dynamic_tinymlp")

    print(f"\n1. Compiling Dynamic TinyMLP to {output_dir} with dynamic_batch=True...")
    compiled = forgetune.compile_model(
        model=model,
        example_inputs=(example_input,),
        output_dir=output_dir,
        autotune=True,
        dynamic_batch=True,
        enable_fusion=True
    )

    print("\n2. Executing across dynamic batch sizes B in [1, 128]...")
    test_batches = [1, 2, 7, 16, 32, 64, 128]
    for b in test_batches:
        x_test = torch.randn(b, 16)
        with torch.no_grad():
            ref_out = model(x_test).numpy()
        
        act_out = compiled.run(x_test)
        max_abs = float(np.max(np.abs(ref_out - act_out)))
        max_rel = float(np.max(np.abs((ref_out - act_out) / (np.abs(ref_out) + 1e-7))))
        
        assert act_out.shape == (b, 8)
        assert np.allclose(ref_out, act_out, atol=1e-4, rtol=1e-4)
        print(f"  [+] Batch B={b:3d}: PASS (max_abs={max_abs:.2e}, max_rel={max_rel:.2e})")

    print("\n3. Standalone microbenchmarking...")
    lat_us = compiled.benchmark_kernel(iters=1000, batch_size=16)
    print(f"  Native kernel latency (B=16): {lat_us:.3f} us")

    print("\n=================================================================")
    print(" Model A (Dynamic TinyMLP) Completed Successfully!")
    print("=================================================================")

if __name__ == "__main__":
    main()
