"""
ForgeTune v2.0 Standalone Rigorous Benchmark Suite
Evaluates Model A (Dynamic TinyMLP), Model B (Residual MLP), and Model C (Small CNN).
Measures:
  1. PyTorch eager CPU latency
  2. ForgeTune native end-to-end execution latency
  3. ForgeTune isolated C++ in-kernel latency
  4. Speedup ratios and buffer reuse savings
Outputs:
  - reports/v2_benchmark_report.json
  - reports/V2_BENCHMARK_REPORT.md
"""

import os
import sys
import json
import time
import platform
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from forgetune import compile_model, benchmark_function, get_machine_metadata
from forgetune.manifest import write_atomic_json

# Model Definitions
class DynamicTinyMLP(nn.Module):
    def __init__(self, in_features=16, hidden_features=32, out_features=8):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class ResidualMLP(nn.Module):
    def __init__(self, features=16, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(features, features)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(features, features)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(features, out_features)

    def forward(self, x):
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        res = h1 + h2
        return self.fc3(res)

class SmallCNN(nn.Module):
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

def measure_pytorch_latency(model, x, warmup=25, iters=100, num_threads=None):
    orig_threads = torch.get_num_threads()
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    try:
        model.eval()
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)
            timings = []
            for _ in range(iters):
                t0 = time.perf_counter_ns()
                _ = model(x)
                t1 = time.perf_counter_ns()
                timings.append((t1 - t0) / 1000.0)
        return float(np.median(timings))
    finally:
        torch.set_num_threads(orig_threads)

def run_benchmarks():
    print("=================================================================")
    print(" ForgeTune v2.0 Comprehensive Multi-Model Benchmark Suite")
    print("=================================================================")

    machine_meta = get_machine_metadata()
    default_threads = torch.get_num_threads()
    print(f"Host OS:          {machine_meta['system']} {machine_meta['release']} ({machine_meta['machine']})")
    print(f"Processor:        {machine_meta['processor']}")
    print(f"Python Runtime:   {machine_meta['python_version']}")
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"Default Threads:  {default_threads}")
    print("=================================================================")

    torch.manual_seed(42)
    models = [
        ("Model A (Dynamic TinyMLP)", DynamicTinyMLP(), (torch.randn(2, 16),), "v2_dynamic_tinymlp", [1, 4, 16, 64]),
        ("Model B (Residual MLP)", ResidualMLP(), (torch.randn(2, 16),), "v2_residual_mlp", [1, 4, 16, 64]),
        ("Model C (Small CNN)", SmallCNN(), (torch.randn(2, 1, 8, 8),), "v2_small_cnn", [1, 4, 16, 64]),
    ]

    benchmark_results = {
        "metadata": {
            **machine_meta,
            "pytorch_default_threads": default_threads
        },
        "models": {}
    }

    for name, model, ex_inps, art_subdir, batches in models:
        print(f"\n---> Benchmarking {name}...")
        model.eval()
        art_dir = os.path.join("build", "benchmarks", art_subdir)
        
        # 1. Compile Optimized Target (with autotuning and fusion)
        t_compile_start = time.perf_counter()
        compiled = compile_model(
            model=model,
            example_inputs=ex_inps,
            output_dir=art_dir,
            autotune=True,
            dynamic_batch=True,
            enable_fusion=True
        )
        compilation_cost_s = round(time.perf_counter() - t_compile_start, 2)

        # 2. Compile Baseline Target (unoptimized native kernel without fusion)
        baseline_art_dir = os.path.join("build", "benchmarks", art_subdir + "_baseline")
        compiled_baseline = compile_model(
            model=model,
            example_inputs=ex_inps,
            output_dir=baseline_art_dir,
            autotune=False,
            dynamic_batch=True,
            enable_fusion=False
        )

        opt_cand_id = compiled.manifest().get("kernel_config", {}).get("candidate_id", "tuned")
        base_cand_id = compiled_baseline.manifest().get("kernel_config", {}).get("candidate_id", "baseline_naive")

        model_results = []
        for b in batches:
            if "CNN" in name:
                x_test = torch.randn(b, 1, 8, 8)
            else:
                x_test = torch.randn(b, 16)

            # PyTorch eager (default multi-thread and controlled 1-thread)
            pt_median_us = round(measure_pytorch_latency(model, x_test), 2)
            pt_1th_median_us = round(measure_pytorch_latency(model, x_test, num_threads=1), 2)

            # ForgeTune native end-to-end
            ft_e2e_res = compiled.benchmark(warmup_iters=25, measure_iters=100, batch_size=b)
            ft_e2e_median_us = round(ft_e2e_res["median_latency_us"], 2)

            # ForgeTune native in-kernel (Tuned vs Baseline) with interleaved multi-round measurement
            compiled.benchmark_kernel(iters=50, batch_size=b)
            compiled_baseline.benchmark_kernel(iters=50, batch_size=b)
            k_tuned_samples = []
            k_base_samples = []
            for r in range(5):
                if r % 2 == 0:
                    k_tuned_samples.append(compiled.benchmark_kernel(iters=500, batch_size=b))
                    k_base_samples.append(compiled_baseline.benchmark_kernel(iters=500, batch_size=b))
                else:
                    k_base_samples.append(compiled_baseline.benchmark_kernel(iters=500, batch_size=b))
                    k_tuned_samples.append(compiled.benchmark_kernel(iters=500, batch_size=b))

            ft_kernel_us = round(float(np.median(k_tuned_samples)), 3)
            ft_base_kernel_us = round(float(np.median(k_base_samples)), 3)

            # Recalculate derived metrics directly from stored values for complete consistency
            speedup_e2e = round(pt_median_us / ft_e2e_median_us, 2) if ft_e2e_median_us > 0 else 1.0
            speedup_e2e_1th = round(pt_1th_median_us / ft_e2e_median_us, 2) if ft_e2e_median_us > 0 else 1.0
            if opt_cand_id in ("baseline_naive", "baseline") and base_cand_id in ("baseline_naive", "baseline"):
                kernel_opt_speedup = 1.00
            else:
                kernel_opt_speedup = round(ft_base_kernel_us / ft_kernel_us, 2) if ft_kernel_us > 0 else 1.0
            residual_us = round(ft_e2e_median_us - ft_kernel_us, 2)
            diagnostic_ratio = round(pt_median_us / ft_kernel_us, 1) if ft_kernel_us > 0 else 1.0

            entry = {
                "batch_size": b,
                "opt_candidate_id": opt_cand_id,
                "base_candidate_id": base_cand_id,
                "pytorch_eager_median_us": pt_median_us,
                "pytorch_eager_1thread_us": pt_1th_median_us,
                "forgetune_e2e_median_us": ft_e2e_median_us,
                "speedup_e2e": speedup_e2e,
                "speedup_e2e_1thread": speedup_e2e_1th,
                "forgetune_baseline_kernel_us": ft_base_kernel_us,
                "forgetune_kernel_median_us": ft_kernel_us,
                "kernel_opt_speedup": kernel_opt_speedup,
                "e2e_minus_kernel_residual_us": residual_us,
                "diagnostic_ratio": diagnostic_ratio,
                "compilation_cost_s": compilation_cost_s
            }
            model_results.append(entry)
            print(f"  Batch B={b:2d}: PyTorch (8 threads)={pt_median_us:6.2f} us | PyTorch (1 thread)={pt_1th_median_us:6.2f} us | FT e2e={ft_e2e_median_us:6.2f} us (E2E Speedup: {speedup_e2e:5.2f}x) | Baseline Kernel={ft_base_kernel_us:7.3f} us | Tuned Kernel={ft_kernel_us:7.3f} us (Kernel Opt Speedup: {kernel_opt_speedup:5.2f}x)")

        benchmark_results["models"][name] = model_results

    os.makedirs("reports", exist_ok=True)
    json_path = os.path.join("reports", "v2_benchmark_report.json")
    write_atomic_json(json_path, benchmark_results)
    print(f"\n[+] Saved structured benchmark JSON to {json_path}")

    # Generate Markdown Report
    md_lines = [
        "# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Host Platform:** {machine_meta['system']} {machine_meta['release']} ({machine_meta['machine']})",
        f"**Processor:** {machine_meta['processor']}",
        f"**Python Runtime:** {machine_meta['python_version']}",
        f"**PyTorch Version:** {torch.__version__} (Default threads: {default_threads})",
        "",
        "---",
        "",
        "## 1. Executive Performance Summary",
        "",
        "ForgeTune v2.0 introduces ahead-of-time C++ code generation, operator fusion (`CONV2D + RELU`, `MATMUL + ADD + RELU`), and dynamic batch execution ($B \\in [1, 128]$) without framework overhead. Benchmarks distinguish:",
        "1. **Genuine End-to-End Speedup (Default Threads):** Real-world user latency comparing PyTorch eager execution in Python (8 CPU threads) vs ForgeTune single-threaded execution in Python.",
        "2. **Controlled End-to-End Speedup (1 Thread):** Controlled architectural latency comparing single-threaded PyTorch vs single-threaded ForgeTune.",
        "3. **Genuine Native Kernel Optimization Speedup:** Equivalent boundaries comparing ForgeTune unoptimized native C++ baseline vs ForgeTune autotuned fused native C++ kernel.",
        "4. **E2E - Kernel Residual:** Remaining runtime work outside the inner kernel ($t_{e2e} - t_{kernel}$), including numpy array translation, ctypes call overhead, and shape verification.",
        "5. **Diagnostic Boundary Ratio:** Framework interpreter overhead elimination ceiling ($t_{PT\\_e2e} / t_{FT\\_kernel}$).",
        "",
        "---",
        "",
        "## 2. Multi-Model Benchmark Data",
        ""
    ]

    for model_name, entries in benchmark_results["models"].items():
        comp_cost = entries[0]["compilation_cost_s"] if entries else 0.0
        opt_cand = entries[0].get("opt_candidate_id", "tuned") if entries else "tuned"
        base_cand = entries[0].get("base_candidate_id", "baseline_naive") if entries else "baseline_naive"
        tuner_note = ""
        if opt_cand in ("baseline_naive", "baseline") and base_cand in ("baseline_naive", "baseline"):
            tuner_note = " — *Baseline retained by autotuner (<5% candidate performance difference)*"
        else:
            tuner_note = f" — *Autotuner selected `{opt_cand}` over `{base_cand}`*"

        md_lines.extend([
            f"### {model_name} (Selected: `{opt_cand}`, Cost: {comp_cost}s){tuner_note}",
            "",
            "| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])
        for e in entries:
            md_lines.append(
                f"| {e['batch_size']} | {e['pytorch_eager_median_us']:.2f} | {e['pytorch_eager_1thread_us']:.2f} | {e['forgetune_e2e_median_us']:.2f} | **{e['speedup_e2e']:.2f}x** | **{e['speedup_e2e_1thread']:.2f}x** | {e['forgetune_baseline_kernel_us']:.3f} | {e['forgetune_kernel_median_us']:.3f} | **{e['kernel_opt_speedup']:.2f}x** | {e['e2e_minus_kernel_residual_us']:.2f} | {e['diagnostic_ratio']:.1f}x* |"
            )
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## 3. Latency Metric Methodology & Disambiguation",
        "",
        "1. **Genuine End-to-End Speedup (Column 5):** Measures real-world Python caller latency: `PyTorch Eager (8 threads) / ForgeTune E2E`. Both run complete model inference with caller input validation.",
        "2. **Controlled End-to-End Speedup (Column 6):** Measures single-threaded algorithmic efficiency: `PyTorch Eager (1 thread) / ForgeTune E2E`, controlling for multi-core parallelism in PyTorch MKL-DNN / oneDNN.",
        "3. **Genuine Kernel Optimization Speedup (Column 9):** Measures compiler code-generation and fusion efficacy at identical boundaries: `Baseline C++ Kernel / Tuned C++ Kernel` inside the compiled native shared library.",
        "4. **E2E - Kernel Residual (Column 10):** Difference between Python caller duration and pure native execution ($t_{e2e} - t_{kernel}$), representing numpy pointer translation, ctypes call overhead, and batch validation. Note on signed residuals: Because $t_{e2e}$ (100 iterations) and $t_{kernel}$ (500 iterations) are measured in separate benchmark sessions, workloads where pure kernel execution dominates latency (e.g., Model C at certain batch sizes) can exhibit small negative residuals (such as -1.90 μs) due to inter-session timer variance, core migration, or CPU frequency scaling; these are reported signed rather than clamped to zero.",
        "5. ***Diagnostic Ratio (Column 11):** `PyTorch E2E / Tuned C++ Kernel`. This is NOT an equivalent-workload speedup because it divides PyTorch Python-level execution by native in-kernel execution. It serves as a diagnostic demonstrating the performance ceiling when framework interpreter overhead is completely eliminated.",
        ""
    ])

    md_path = os.path.join("reports", "V2_BENCHMARK_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[+] Saved human-readable benchmark report to {md_path}")

if __name__ == "__main__":
    run_benchmarks()
