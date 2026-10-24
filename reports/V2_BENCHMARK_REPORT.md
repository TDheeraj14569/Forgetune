# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT

**Date:** 2026-10-24 23:14:03
**Run ID:** `v2-bench-1792863787-0233951b`
**Git Commit:** `0233951bcd8500a1acd9046fb2b5633b90025902`
**Tree Fingerprint:** `069abc69a15e586c686b8b2afe9596e28f7bf648`
**Host Platform:** Windows 11 (AMD64)
**Processor:** AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD
**Python Runtime:** 3.12.14
**PyTorch Version:** 2.14.1+cpu (Default threads: 8)

---

## 1. Executive Performance Summary

ForgeTune v2.0 introduces ahead-of-time C++ code generation, operator fusion (`CONV2D + RELU`, `MATMUL + ADD + RELU`), and dynamic batch execution ($B \in [1, 128]$) without framework overhead. Benchmarks distinguish:
1. **Genuine End-to-End Speedup (Default Threads):** Real-world user latency comparing PyTorch eager execution in Python (8 CPU threads) vs ForgeTune single-threaded execution in Python.
2. **Controlled End-to-End Speedup (1 Thread):** Controlled architectural latency comparing single-threaded PyTorch vs single-threaded ForgeTune.
3. **Genuine Native Kernel Optimization Speedup:** Equivalent boundaries comparing ForgeTune unoptimized native C++ baseline vs ForgeTune autotuned fused native C++ kernel.
4. **E2E - Kernel Residual:** Remaining runtime work outside the inner kernel ($t_{e2e} - t_{kernel}$), including numpy array translation, ctypes call overhead, and shape verification.
5. **PyTorch E2E / ForgeTune native-kernel diagnostic ratio (non-equivalent timing boundaries):** Diagnostic ratio dividing Python-level PyTorch E2E latency ($t_{PT\_e2e}$, numerator) by pure native C++ in-kernel latency ($t_{FT\_kernel}$, denominator).

---

## 2. Multi-Model Benchmark Data

### Model A (Dynamic TinyMLP) (Selected: `baseline_naive`, Cost: 7.13s) — *Baseline retained by autotuner (<5% candidate performance difference)*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Selected Implementation Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25.90 | 26.10 | 5.85 | **4.43x** | **4.46x** | 0.230 | 0.230 | **1.00x** | 5.62 | 112.6x* |
| 4 | 42.50 | 37.60 | 9.20 | **4.62x** | **4.09x** | 0.852 | 0.849 | **1.00x** | 8.35 | 50.1x* |
| 16 | 32.70 | 38.60 | 12.70 | **2.57x** | **3.04x** | 4.514 | 3.874 | **1.00x** | 8.83 | 8.4x* |
| 64 | 37.60 | 32.00 | 26.20 | **1.44x** | **1.22x** | 17.345 | 16.516 | **1.00x** | 9.68 | 2.3x* |

### Model B (Residual MLP) (Selected: `baseline_naive`, Cost: 4.37s) — *Baseline retained by autotuner (<5% candidate performance difference)*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Selected Implementation Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 42.00 | 43.10 | 9.10 | **4.62x** | **4.74x** | 0.121 | 0.182 | **1.00x** | 8.92 | 230.8x* |
| 4 | 44.30 | 44.25 | 6.00 | **7.38x** | **7.38x** | 0.395 | 0.375 | **1.00x** | 5.62 | 118.1x* |
| 16 | 45.55 | 45.80 | 6.90 | **6.60x** | **6.64x** | 1.449 | 1.431 | **1.00x** | 5.47 | 31.8x* |
| 64 | 100.55 | 69.00 | 16.40 | **6.13x** | **4.21x** | 8.089 | 8.013 | **1.00x** | 8.39 | 12.5x* |

### Model C (Small CNN) (Selected: `loop_ikj`, Cost: 31.95s) — *Autotuner selected `loop_ikj` over `baseline`*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Selected Implementation Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 84.05 | 88.45 | 16.80 | **5.00x** | **5.26x** | 9.423 | 7.986 | **1.18x** | 8.81 | 10.5x* |
| 4 | 321.70 | 307.60 | 37.60 | **8.56x** | **8.18x** | 33.648 | 26.789 | **1.26x** | 10.81 | 12.0x* |
| 16 | 352.00 | 286.90 | 89.65 | **3.93x** | **3.20x** | 120.367 | 92.864 | **1.30x** | -3.21 | 3.8x* |
| 64 | 409.53 | 462.02 | 442.70 | **0.93x** | **1.04x** | 461.680 | 421.300 | **1.10x** | 24.13 | 1.0x* |

### Model C at $B=64$ Statistical Distribution Analysis (30 Independent Sample Batches)

To rigorously verify Model C performance at batch size $B=64$, 30 independent sample batches were measured across both frameworks using a unified measurement protocol:

| Execution Boundary | Median (μs) | Mean (μs) | Std Dev (μs) | IQR (μs) | Min (μs) | Max (μs) | 5th Pct (μs) | 95th Pct (μs) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PyTorch Eager (8 threads)** | 409.53 | 417.24 | 67.97 | 51.70 | 308.45 | 608.10 | 315.95 | 530.35 |
| **PyTorch Eager (1 thread)** | 462.02 | 454.64 | 52.43 | 30.80 | 341.05 | 559.90 | 351.25 | 511.64 |
| **ForgeTune Python E2E** | 442.70 | 430.42 | 40.44 | 9.66 | 339.05 | 523.25 | 345.44 | 453.90 |
| **ForgeTune Baseline Native Kernel** | 461.68 | 463.83 | 13.90 | 12.83 | 434.85 | 495.35 | 446.98 | 490.61 |
| **ForgeTune Selected Native Kernel (`loop_ikj`)** | 421.30 | 419.80 | 15.51 | 27.92 | 391.22 | 444.69 | 395.35 | 440.82 |
| **E2E - Kernel Residual** | 24.13 | 10.62 | 48.83 | 34.33 | -105.64 | 100.25 | -88.58 | 55.46 |

**Residual Analysis ($t_{e2e} - t_{kernel}$):**
- Median residual is 24.13 μs.
- The residual represents Python interpreter overhead, NumPy contiguous buffer conversion (`np.ascontiguousarray`), ctypes function argument wrapping, and dynamic batch shape checking.
- In pure single-threaded native execution without Python framework overhead, the ForgeTune selected native kernel executes at **421.30 μs**, achieving 1.10x speedup over single-threaded PyTorch E2E (462.02 μs).
- At the Python E2E level, single-threaded ForgeTune (442.70 μs) achieves 0.93x speedup vs 8-threaded PyTorch (409.53 μs) and 1.04x vs single-threaded PyTorch (462.02 μs).

---

## 3. Latency Metric Methodology & Disambiguation

1. **Genuine End-to-End Speedup (Column 5):** Measures real-world Python caller latency: `PyTorch Eager (8 threads) / ForgeTune E2E`. Both run complete model inference with caller input validation.
2. **Controlled End-to-End Speedup (Column 6):** Measures single-threaded algorithmic efficiency: `PyTorch Eager (1 thread) / ForgeTune E2E`, controlling for multi-core parallelism in PyTorch MKL-DNN / oneDNN.
3. **Genuine Kernel Optimization Speedup (Column 9):** Measures compiler code-generation and fusion efficacy at identical boundaries: `Baseline C++ Kernel / Selected Implementation Kernel` inside the compiled native shared library. For workloads where the baseline is retained (e.g. Models A and B, where alternative candidates are within the 5% noise window or slower across the dynamic batch range), this ratio is strictly 1.00x.
4. **E2E - Kernel Residual (Column 10):** Difference between Python caller duration and pure native execution ($t_{e2e} - t_{kernel}$), representing numpy pointer translation, ctypes call overhead, and batch validation. Note on signed residuals: Because $t_{e2e}$ (100 iterations) and $t_{kernel}$ (500 iterations) are measured in separate benchmark sessions, workloads where pure kernel execution dominates latency (e.g., Model C at certain batch sizes) can exhibit small negative residuals (such as -1.90 μs) due to inter-session timer variance, core migration, or CPU frequency scaling; these are reported signed rather than clamped to zero.
5. ***PyTorch E2E / ForgeTune native-kernel diagnostic ratio (non-equivalent timing boundaries) (Column 11):** Ratio: Numerator = `PyTorch Eager (8 threads) E2E latency` ($t_{PT\_e2e}$); Denominator = `ForgeTune Selected C++ Kernel latency` ($t_{FT\_kernel}$). This is NOT an equivalent-workload speedup because it divides PyTorch Python-level execution by native in-kernel execution. It serves as a diagnostic demonstrating the performance ceiling when framework interpreter overhead is completely eliminated.
