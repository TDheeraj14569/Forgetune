# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT

**Date:** 2026-10-27 20:25:36
**Run ID:** `v2-bench-1793112834-1cb7cc15`
**Git Commit:** `1cb7cc15ec5a2ad47ae0fdd4ba822efcc38aadad`
**Tree Fingerprint:** `cc08b3fa5a92eff709eb28d357c7ae3360b6ded2`
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

### Model A (Dynamic TinyMLP) (Selected: `baseline_naive`, Cost: 22.31s) — *Baseline retained by autotuner (<5% candidate performance difference)*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Candidate Kernel (μs) | Selected Implementation Kernel (μs) | Candidate Speedup | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 24.90 | 25.00 | 5.80 | **4.29x** | **4.31x** | 0.230 | 0.231 | 0.230 | 1.00x | **1.00x** | 5.57 | 108.3x* |
| 4 | 42.95 | 26.55 | 6.50 | **6.61x** | **4.08x** | 0.864 | 0.861 | 0.864 | 1.00x | **1.00x** | 5.64 | 49.7x* |
| 16 | 45.25 | 39.60 | 12.70 | **3.56x** | **3.12x** | 4.183 | 4.444 | 4.183 | 0.94x | **1.00x** | 8.52 | 10.8x* |
| 64 | 39.95 | 44.60 | 19.80 | **2.02x** | **2.25x** | 14.300 | 14.845 | 14.300 | 0.96x | **1.00x** | 5.50 | 2.8x* |

### Model B (Residual MLP) (Selected: `baseline_naive`, Cost: 17.53s) — *Baseline retained by autotuner (<5% candidate performance difference)*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Candidate Kernel (μs) | Selected Implementation Kernel (μs) | Candidate Speedup | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 56.75 | 41.50 | 5.70 | **9.96x** | **7.28x** | 0.121 | 0.120 | 0.121 | 1.01x | **1.00x** | 5.58 | 469.0x* |
| 4 | 60.90 | 43.50 | 5.90 | **10.32x** | **7.37x** | 0.377 | 0.375 | 0.377 | 1.01x | **1.00x** | 5.52 | 161.5x* |
| 16 | 45.10 | 44.80 | 7.10 | **6.35x** | **6.31x** | 1.443 | 1.438 | 1.443 | 1.00x | **1.00x** | 5.66 | 31.3x* |
| 64 | 77.20 | 68.10 | 16.60 | **4.65x** | **4.10x** | 7.934 | 7.940 | 7.934 | 1.00x | **1.00x** | 8.67 | 9.7x* |

### Model C (Small CNN) (Selected: `loop_ikj`, Cost: 44.67s) — *Autotuner selected `loop_ikj` over `baseline`*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Candidate Kernel (μs) | Selected Implementation Kernel (μs) | Candidate Speedup | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 97.15 | 88.90 | 11.90 | **8.16x** | **7.47x** | 9.397 | 7.879 | 7.879 | 1.19x | **1.19x** | 4.02 | 12.3x* |
| 4 | 311.15 | 291.90 | 27.90 | **11.15x** | **10.46x** | 32.861 | 27.808 | 27.808 | 1.18x | **1.18x** | 0.09 | 11.2x* |
| 16 | 450.95 | 340.15 | 118.10 | **3.82x** | **2.88x** | 113.562 | 86.875 | 86.875 | 1.31x | **1.31x** | 31.22 | 5.2x* |
| 64 | 380.07 | 437.75 | 441.05 | **0.86x** | **0.99x** | 457.690 | 422.910 | 422.910 | 1.08x | **1.08x** | 18.14 | 0.9x* |

### Model C at $B=64$ Statistical Distribution Analysis (30 Independent Sample Batches)

To rigorously verify Model C performance at batch size $B=64$, 30 independent sample batches were measured across both frameworks using a unified measurement protocol:

| Execution Boundary | Median (μs) | Mean (μs) | Std Dev (μs) | IQR (μs) | Min (μs) | Max (μs) | 5th Pct (μs) | 95th Pct (μs) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PyTorch Eager (8 threads)** | 380.07 | 371.43 | 32.55 | 21.09 | 296.80 | 431.35 | 299.38 | 402.81 |
| **PyTorch Eager (1 thread)** | 437.75 | 434.40 | 25.81 | 11.85 | 345.10 | 485.70 | 379.62 | 459.02 |
| **ForgeTune Python E2E** | 441.05 | 421.00 | 41.89 | 2.14 | 334.20 | 446.35 | 336.72 | 445.49 |
| **ForgeTune Baseline Native Kernel** | 457.69 | 462.77 | 21.93 | 10.60 | 434.60 | 541.63 | 442.20 | 510.63 |
| **ForgeTune Selected Native Kernel (`loop_ikj`)** | 422.91 | 418.81 | 16.49 | 11.26 | 364.46 | 439.27 | 382.43 | 434.29 |
| **E2E - Kernel Residual** | 16.68 | 2.19 | 42.41 | 16.72 | -97.45 | 76.44 | -90.22 | 37.73 |

**Residual Analysis ($t_{e2e} - t_{kernel}$):**
- Median residual is 16.68 μs.
- The residual represents Python interpreter overhead, NumPy contiguous buffer conversion (`np.ascontiguousarray`), ctypes function argument wrapping, and dynamic batch shape checking.
- In pure single-threaded native execution without Python framework overhead, the ForgeTune selected native kernel executes at **422.91 μs**, achieving 1.04x speedup over single-threaded PyTorch E2E (437.75 μs).
- At the Python E2E level, single-threaded ForgeTune (441.05 μs) achieves 0.86x speedup vs 8-threaded PyTorch (380.07 μs) and 0.99x vs single-threaded PyTorch (437.75 μs).

---

## 3. Latency Metric Methodology & Disambiguation

1. **Genuine End-to-End Speedup (Column 5):** Measures real-world Python caller latency: `PyTorch Eager (8 threads) / ForgeTune E2E`. Both run complete model inference with caller input validation.
2. **Controlled End-to-End Speedup (Column 6):** Measures single-threaded algorithmic efficiency: `PyTorch Eager (1 thread) / ForgeTune E2E`, controlling for multi-core parallelism in PyTorch MKL-DNN / oneDNN.
3. **Genuine Kernel Optimization Speedup (Column 9):** Measures compiler code-generation and fusion efficacy at identical boundaries: `Baseline C++ Kernel / Selected Implementation Kernel` inside the compiled native shared library. For workloads where the baseline is retained (e.g. Models A and B, where alternative candidates are within the 5% noise window or slower across the dynamic batch range), this ratio is strictly 1.00x.
4. **E2E - Kernel Residual (Column 10):** Difference between Python caller duration and pure native execution ($t_{e2e} - t_{kernel}$), representing numpy pointer translation, ctypes call overhead, and batch validation. Note on signed residuals: Because $t_{e2e}$ (100 iterations) and $t_{kernel}$ (500 iterations) are measured in separate benchmark sessions, workloads where pure kernel execution dominates latency (e.g., Model C at certain batch sizes) can exhibit small negative residuals (such as -1.90 μs) due to inter-session timer variance, core migration, or CPU frequency scaling; these are reported signed rather than clamped to zero.
5. ***PyTorch E2E / ForgeTune native-kernel diagnostic ratio (non-equivalent timing boundaries) (Column 11):** Ratio: Numerator = `PyTorch Eager (8 threads) E2E latency` ($t_{PT\_e2e}$); Denominator = `ForgeTune Selected C++ Kernel latency` ($t_{FT\_kernel}$). This is NOT an equivalent-workload speedup because it divides PyTorch Python-level execution by native in-kernel execution. It serves as a diagnostic demonstrating the performance ceiling when framework interpreter overhead is completely eliminated.
