"""
ForgeTune Example: TinyMLP End-to-End Pipeline
Demonstrates:
  1. Defining TinyMLP (4 -> 8 -> 3 with ReLU)
  2. Exporting with torch.export
  3. Compiling with ForgeTune self-optimizing engine
  4. Running native inference without PyTorch arithmetic
  5. Numerical comparison against PyTorch reference
  6. Displaying autotuning and benchmark results
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
import forgetune

class TinyMLP(nn.Module):
    """
    Two-layer Multi-Layer Perceptron:
      Layer 1: Linear(4, 8) + ReLU
      Layer 2: Linear(8, 3)
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def main():
    print("=================================================================")
    print(" ForgeTune: Compiling TinyMLP Model (End-to-End)")
    print("=================================================================")

    # 1. Deterministic model setup
    torch.manual_seed(42)
    model = TinyMLP().eval()

    # Fixed example input: batch size 2, input dimension 4
    x = torch.randn(2, 4)

    output_dir = os.path.join("build", "artifacts", "tiny_mlp")
    print(f"\n1. Compiling model with autotuning to {output_dir}...")
    compiled = forgetune.compile_model(
        model=model,
        example_inputs=(x,),
        output_dir=output_dir,
        autotune=True,
        max_autotune_budget=8
    )

    print("\n2. Executing native C++ CPU inference...")
    native_result = compiled.run(x)

    print("\n3. Evaluating independent PyTorch reference...")
    with torch.no_grad():
        torch_result = model(x).numpy()

    print("\n4. Numerical Verification:")
    print("Native output:\n", native_result)
    print("PyTorch output:\n", torch_result)

    max_abs_diff = float(np.max(np.abs(native_result - torch_result)))
    max_rel_diff = float(np.max(np.abs((native_result - torch_result) / (np.abs(torch_result) + 1e-7))))
    print(f"Max absolute difference: {max_abs_diff:.8e}")
    print(f"Max relative difference: {max_rel_diff:.8e}")
    assert max_abs_diff < 1e-4, f"Numerical mismatch exceeds tolerance: {max_abs_diff}"
    print(">> NUMERICAL VERIFICATION PASSED <<")

    print("\n5. Optimization and Autotuning Report:")
    report = compiled.optimization_report()
    selected = report.get("selected_candidate", {})
    print(f"Selected Candidate: {selected.get('candidate_id')}")
    print(f"  Variant:    {selected.get('variant')}")
    print(f"  Fusion:     {selected.get('enable_fusion')}")
    tile_sizes = selected.get("tile_sizes")
    tile_str = f"({tile_sizes[0]}, {tile_sizes[1]}, {tile_sizes[2]})" if tile_sizes else "N/A"
    print(f"  Tile sizes: {tile_str}")
    print(f"  Compiler:   {selected.get('opt_level')}")

    if "best_median_latency_us" in report:
        print("\nMeasured Latency (Interleaved Candidate Evaluation):")
        print(f"  Selected candidate (e2e): {report['best_median_latency_us']:.2f} us")
        if report.get("baseline_median_latency_us"):
            print(f"  Baseline naive (e2e):     {report['baseline_median_latency_us']:.2f} us")
            print(f"  End-to-End Latency Ratio: {report['measured_speedup']:.2f}x (within ctypes dispatch noise)")
        if report.get("selection_rationale"):
            print(f"  Selection Rationale:      {report['selection_rationale']}")

    print("\n6. Isolated Pure In-Kernel Execution (10,000 iterations in C++ via steady_clock)...")
    k_lat = compiled.benchmark_kernel(iters=10000)
    print(f"  In-Kernel Latency: {k_lat:.4f} us ({k_lat*1000.0:.1f} ns)")

    print("\n7. End-to-End Python Harness (100 measured iterations)...")
    bench_results = compiled.benchmark(warmup_iters=20, measure_iters=100)
    print(f"  Median Latency: {bench_results['median_latency_us']:.2f} us")
    print(f"  Min Latency:    {bench_results['min_latency_us']:.2f} us")
    print(f"  Max Latency:    {bench_results['max_latency_us']:.2f} us")
    print(f"  Std Dev:        {bench_results['std_dev_us']:.2f} us")
    print(f"  IQR:            {bench_results['iqr_us']:.2f} us")

    print("\n=================================================================")
    print(" ForgeTune TinyMLP Execution Completed Successfully!")
    print("=================================================================")

if __name__ == "__main__":
    main()
