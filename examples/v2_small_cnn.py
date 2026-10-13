"""
ForgeTune v2.0 Example: Model C — Small CNN End-to-End Pipeline
Demonstrates:
  1. Defining Small CNN (Conv2D -> ReLU -> Conv2D -> ReLU -> Flatten -> Linear)
  2. Ahead-of-time export with dynamic batch dimension B in [1, 128]
  3. Compiling with ForgeTune self-optimizing engine and operator fusion
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

class SmallCNN(nn.Module):
    """
    Small Convolutional Neural Network:
      conv1: Conv2d(1 -> 8, kernel_size=3, stride=1, padding=1)
      act1:  ReLU()
      conv2: Conv2d(8 -> 16, kernel_size=3, stride=2, padding=1)
      act2:  ReLU()
      flat:  Flatten(start_dim=1)
      fc:    Linear(16 * 4 * 4 -> 10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        return self.fc(x)

def main():
    print("=================================================================")
    print(" ForgeTune v2.0: Model C — Small CNN End-to-End")
    print("=================================================================")

    torch.manual_seed(42)
    model = SmallCNN().eval()

    example_input = torch.randn(2, 1, 8, 8)
    output_dir = os.path.join("build", "artifacts", "v2_small_cnn")

    print(f"\n1. Compiling Small CNN to {output_dir} with dynamic_batch=True...")
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
        x_test = torch.randn(b, 1, 8, 8)
        with torch.no_grad():
            ref_out = model(x_test).numpy()
        
        act_out = compiled.run(x_test)
        max_abs = float(np.max(np.abs(ref_out - act_out)))
        max_rel = float(np.max(np.abs((ref_out - act_out) / (np.abs(ref_out) + 1e-7))))
        
        assert act_out.shape == (b, 10)
        assert np.allclose(ref_out, act_out, atol=1e-4, rtol=1e-4)
        print(f"  [+] Batch B={b:3d}: PASS (max_abs={max_abs:.2e}, max_rel={max_rel:.2e})")

    print("\n3. Standalone microbenchmarking...")
    lat_us = compiled.benchmark_kernel(iters=500, batch_size=16)
    print(f"  Native kernel latency (B=16): {lat_us:.3f} us")

    print("\n=================================================================")
    print(" Model C (Small CNN) Completed Successfully!")
    print("=================================================================")

if __name__ == "__main__":
    main()
