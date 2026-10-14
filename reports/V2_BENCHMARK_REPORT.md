# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT

**Date:** 2026-10-15 00:27:56
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
5. **Diagnostic Boundary Ratio:** Framework interpreter overhead elimination ceiling ($t_{PT\_e2e} / t_{FT\_kernel}$).

---

## 2. Multi-Model Benchmark Data

### Model A (Dynamic TinyMLP) (Selected: `baseline_naive`, Cost: 3.28s) — *Baseline retained by autotuner (<5% candidate performance difference)*

| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 38.80 | 24.70 | 8.30 | **4.67x** | **2.98x** | 0.280 | 0.276 | **1.00x** | 8.02 | 140.6x* |
| 4 | 44.00 | 37.20 | 12.80 | **3.44x** | **2.91x** | 1.022 | 1.023 | **1.00x** | 11.78 | 43.0x* |
| 16 | 44.95 | 38.50 | 18.20 | **2.47x** | **2.12x** | 4.007 | 4.077 | **1.00x** | 14.12 | 11.0x* |
| 64 | 37.60 | 32.05 | 28.50 | **1.32x** | **1.12x** | 15.693 | 15.382 | **1.00x** | 13.12 | 2.4x* |

### Model B (Residual MLP) (Selected: `baseline_naive`, Cost: 6.73s) — *Baseline retained by autotuner (<5% candidate performance difference)*

| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 42.00 | 41.60 | 9.00 | **4.67x** | **4.62x** | 0.117 | 0.116 | **1.00x** | 8.88 | 362.1x* |
| 4 | 44.45 | 43.80 | 9.40 | **4.73x** | **4.66x** | 0.357 | 0.355 | **1.00x** | 9.04 | 125.2x* |
| 16 | 44.70 | 44.60 | 9.60 | **4.66x** | **4.65x** | 1.400 | 1.349 | **1.00x** | 8.25 | 33.1x* |
| 64 | 76.00 | 67.80 | 14.50 | **5.24x** | **4.68x** | 7.282 | 6.308 | **1.00x** | 8.19 | 12.0x* |

### Model C (Small CNN) (Selected: `loop_ikj`, Cost: 11.37s) — *Autotuner selected `loop_ikj` over `baseline`*

| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 72.60 | 88.70 | 22.00 | **3.30x** | **4.03x** | 9.595 | 8.656 | **1.11x** | 13.34 | 8.4x* |
| 4 | 320.05 | 390.65 | 49.50 | **6.47x** | **7.89x** | 39.511 | 36.079 | **1.10x** | 13.42 | 8.9x* |
| 16 | 370.90 | 324.95 | 170.90 | **2.17x** | **1.90x** | 157.571 | 136.160 | **1.16x** | 34.74 | 2.7x* |
| 64 | 435.25 | 349.00 | 590.75 | **0.74x** | **0.59x** | 644.734 | 539.183 | **1.20x** | 51.57 | 0.8x* |

---

## 3. Latency Metric Methodology & Disambiguation

1. **Genuine End-to-End Speedup (Column 5):** Measures real-world Python caller latency: `PyTorch Eager (8 threads) / ForgeTune E2E`. Both run complete model inference with caller input validation.
2. **Controlled End-to-End Speedup (Column 6):** Measures single-threaded algorithmic efficiency: `PyTorch Eager (1 thread) / ForgeTune E2E`, controlling for multi-core parallelism in PyTorch MKL-DNN / oneDNN.
3. **Genuine Kernel Optimization Speedup (Column 9):** Measures compiler code-generation and fusion efficacy at identical boundaries: `Baseline C++ Kernel / Tuned C++ Kernel` inside the compiled native shared library.
4. **E2E - Kernel Residual (Column 10):** Difference between Python caller duration and pure native execution ($t_{e2e} - t_{kernel}$), representing numpy pointer translation, ctypes call overhead, and batch validation. Note on signed residuals: Because $t_{e2e}$ (100 iterations) and $t_{kernel}$ (500 iterations) are measured in separate benchmark sessions, workloads where pure kernel execution dominates latency (e.g., Model C at certain batch sizes) can exhibit small negative residuals (such as -1.90 μs) due to inter-session timer variance, core migration, or CPU frequency scaling; these are reported signed rather than clamped to zero.
5. ***Diagnostic Ratio (Column 11):** `PyTorch E2E / Tuned C++ Kernel`. This is NOT an equivalent-workload speedup because it divides PyTorch Python-level execution by native in-kernel execution. It serves as a diagnostic demonstrating the performance ceiling when framework interpreter overhead is completely eliminated.
