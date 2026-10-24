# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT

**Date:** 2026-10-24 17:40:38
**Run ID:** `v2-bench-1792843776-9330aa2d`
**Git Commit:** `9330aa2de59ebc8bd818e3d85c8430f4b513dee0`
**Tree Fingerprint:** `2c5df450c6184e27f4efa3ec32e35b36309887f6`
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

### Model A (Dynamic TinyMLP) (Selected: `baseline_naive`, Cost: 8.1s) — *Baseline retained by autotuner (<5% candidate performance difference)*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Selected Implementation Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 29.25 | 24.80 | 5.90 | **4.96x** | **4.20x** | 0.317 | 0.319 | **1.00x** | 5.58 | 91.7x* |
| 4 | 62.55 | 38.20 | 9.80 | **6.38x** | **3.90x** | 1.159 | 1.158 | **1.00x** | 8.64 | 54.0x* |
| 16 | 49.75 | 39.15 | 12.80 | **3.89x** | **3.06x** | 4.496 | 4.505 | **1.00x** | 8.30 | 11.0x* |
| 64 | 50.95 | 44.95 | 28.00 | **1.82x** | **1.61x** | 18.071 | 17.930 | **1.00x** | 10.07 | 2.8x* |

### Model B (Residual MLP) (Selected: `baseline_naive`, Cost: 4.45s) — *Baseline retained by autotuner (<5% candidate performance difference)*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Selected Implementation Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 41.80 | 65.80 | 9.75 | **4.29x** | **6.75x** | 0.168 | 0.169 | **1.00x** | 9.58 | 247.3x* |
| 4 | 68.25 | 69.95 | 10.30 | **6.63x** | **6.79x** | 0.609 | 0.570 | **1.00x** | 9.73 | 119.7x* |
| 16 | 45.25 | 46.50 | 7.00 | **6.46x** | **6.64x** | 1.934 | 2.088 | **1.00x** | 4.91 | 21.7x* |
| 64 | 223.50 | 70.55 | 15.90 | **14.06x** | **4.44x** | 8.009 | 8.306 | **1.00x** | 7.59 | 26.9x* |

### Model C (Small CNN) (Selected: `loop_ikj`, Cost: 35.43s) — *Autotuner selected `loop_ikj` over `baseline`*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Selected Implementation Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 133.90 | 104.10 | 18.30 | **7.32x** | **5.69x** | 9.889 | 8.348 | **1.18x** | 9.95 | 16.0x* |
| 4 | 480.40 | 357.25 | 37.50 | **12.81x** | **9.53x** | 34.279 | 28.864 | **1.19x** | 8.64 | 16.6x* |
| 16 | 347.50 | 335.55 | 118.30 | **2.94x** | **2.84x** | 132.560 | 110.579 | **1.20x** | 7.72 | 3.1x* |
| 64 | 489.42 | 493.68 | 450.68 | **1.09x** | **1.10x** | 521.310 | 439.830 | **1.19x** | 11.50 | 1.1x* |

### Model C at $B=64$ Statistical Distribution Analysis (30 Independent Sample Batches)

To rigorously verify Model C performance at batch size $B=64$, 30 independent sample batches were measured across both frameworks using a unified measurement protocol:

| Execution Boundary | Median (μs) | Mean (μs) | Std Dev (μs) | IQR (μs) | Min (μs) | Max (μs) | 5th Pct (μs) | 95th Pct (μs) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PyTorch Eager (8 threads)** | 489.42 | 537.24 | 110.16 | 184.46 | 408.75 | 752.10 | 409.88 | 716.45 |
| **PyTorch Eager (1 thread)** | 493.68 | 500.68 | 45.49 | 48.44 | 394.45 | 621.45 | 448.52 | 582.51 |
| **ForgeTune Python E2E** | 450.68 | 456.05 | 16.15 | 16.71 | 440.75 | 515.20 | 442.32 | 482.60 |
| **ForgeTune Baseline Native Kernel** | 521.31 | 521.96 | 18.86 | 24.56 | 480.22 | 564.08 | 491.95 | 551.14 |
| **ForgeTune Selected Native Kernel (`loop_ikj`)** | 439.83 | 441.79 | 15.02 | 14.63 | 399.56 | 474.06 | 424.02 | 468.16 |
| **E2E - Kernel Residual** | 11.50 | 14.26 | 22.92 | 25.44 | -28.11 | 87.34 | -15.34 | 51.80 |

**Residual Analysis ($t_{e2e} - t_{kernel}$):**
- Median residual is 11.50 μs.
- The residual represents Python interpreter overhead, NumPy contiguous buffer conversion (`np.ascontiguousarray`), ctypes function argument wrapping, and dynamic batch shape checking.
- In pure single-threaded native execution without Python framework overhead, the ForgeTune selected native kernel executes at **439.83 μs**, achieving 1.12x speedup over single-threaded PyTorch E2E (493.68 μs).
- At the Python E2E level, single-threaded ForgeTune (450.68 μs) achieves 1.09x speedup vs 8-threaded PyTorch (489.42 μs) and 1.10x vs single-threaded PyTorch (493.68 μs).

---

## 3. Latency Metric Methodology & Disambiguation

1. **Genuine End-to-End Speedup (Column 5):** Measures real-world Python caller latency: `PyTorch Eager (8 threads) / ForgeTune E2E`. Both run complete model inference with caller input validation.
2. **Controlled End-to-End Speedup (Column 6):** Measures single-threaded algorithmic efficiency: `PyTorch Eager (1 thread) / ForgeTune E2E`, controlling for multi-core parallelism in PyTorch MKL-DNN / oneDNN.
3. **Genuine Kernel Optimization Speedup (Column 9):** Measures compiler code-generation and fusion efficacy at identical boundaries: `Baseline C++ Kernel / Selected Implementation Kernel` inside the compiled native shared library. For workloads where the baseline is retained (e.g. Models A and B, where alternative candidates are within the 5% noise window or slower across the dynamic batch range), this ratio is strictly 1.00x.
4. **E2E - Kernel Residual (Column 10):** Difference between Python caller duration and pure native execution ($t_{e2e} - t_{kernel}$), representing numpy pointer translation, ctypes call overhead, and batch validation. Note on signed residuals: Because $t_{e2e}$ (100 iterations) and $t_{kernel}$ (500 iterations) are measured in separate benchmark sessions, workloads where pure kernel execution dominates latency (e.g., Model C at certain batch sizes) can exhibit small negative residuals (such as -1.90 μs) due to inter-session timer variance, core migration, or CPU frequency scaling; these are reported signed rather than clamped to zero.
5. ***PyTorch E2E / ForgeTune native-kernel diagnostic ratio (non-equivalent timing boundaries) (Column 11):** Ratio: Numerator = `PyTorch Eager (8 threads) E2E latency` ($t_{PT\_e2e}$); Denominator = `ForgeTune Selected C++ Kernel latency` ($t_{FT\_kernel}$). This is NOT an equivalent-workload speedup because it divides PyTorch Python-level execution by native in-kernel execution. It serves as a diagnostic demonstrating the performance ceiling when framework interpreter overhead is completely eliminated.
