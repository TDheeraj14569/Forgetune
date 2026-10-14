# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT

**Date:** 2026-10-14 22:59:22
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

### Model A (Dynamic TinyMLP) (Compilation Cost: 3.52s)

| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 39.50 | 24.90 | 14.00 | **2.82x** | **1.78x** | 0.298 | 0.294 | **1.01x** | 13.71 | 134.4x* |
| 4 | 43.25 | 39.95 | 13.40 | **3.23x** | **2.98x** | 1.238 | 1.025 | **1.21x** | 12.38 | 42.2x* |
| 16 | 49.95 | 40.10 | 16.50 | **3.03x** | **2.43x** | 3.975 | 4.020 | **0.99x** | 12.48 | 12.4x* |
| 64 | 49.70 | 50.10 | 28.10 | **1.77x** | **1.78x** | 14.277 | 15.779 | **0.90x** | 12.32 | 3.1x* |

### Model B (Residual MLP) (Compilation Cost: 6.22s)

| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 41.90 | 41.80 | 8.60 | **4.87x** | **4.86x** | 0.115 | 0.116 | **0.99x** | 8.48 | 361.2x* |
| 4 | 43.60 | 43.90 | 8.50 | **5.13x** | **5.16x** | 0.354 | 0.358 | **0.99x** | 8.14 | 121.8x* |
| 16 | 45.30 | 45.30 | 10.80 | **4.19x** | **4.19x** | 1.342 | 1.397 | **0.96x** | 9.40 | 32.4x* |
| 64 | 76.90 | 48.90 | 20.40 | **3.77x** | **2.40x** | 7.300 | 7.297 | **1.00x** | 13.10 | 10.5x* |

### Model C (Small CNN) (Compilation Cost: 14.34s)

| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 73.00 | 87.90 | 29.90 | **2.44x** | **2.94x** | 18.475 | 17.079 | **1.08x** | 12.82 | 4.3x* |
| 4 | 335.50 | 395.20 | 82.65 | **4.06x** | **4.78x** | 73.840 | 68.319 | **1.08x** | 14.33 | 4.9x* |
| 16 | 372.80 | 345.35 | 285.35 | **1.31x** | **1.21x** | 292.381 | 277.160 | **1.05x** | 8.19 | 1.3x* |
| 64 | 411.80 | 476.90 | 1117.25 | **0.37x** | **0.43x** | 1065.557 | 927.301 | **1.15x** | 189.95 | 0.4x* |

---

## 3. Latency Metric Methodology & Disambiguation

1. **Genuine End-to-End Speedup (Column 5):** Measures real-world Python caller latency: `PyTorch Eager (8 threads) / ForgeTune E2E`. Both run complete model inference with caller input validation.
2. **Controlled End-to-End Speedup (Column 6):** Measures single-threaded algorithmic efficiency: `PyTorch Eager (1 thread) / ForgeTune E2E`, controlling for multi-core parallelism in PyTorch MKL-DNN / oneDNN.
3. **Genuine Kernel Optimization Speedup (Column 9):** Measures compiler code-generation and fusion efficacy at identical boundaries: `Baseline C++ Kernel / Tuned C++ Kernel` inside the compiled native shared library.
4. **E2E - Kernel Residual (Column 10):** Difference between Python caller duration and pure native execution ($t_{e2e} - t_{kernel}$), representing numpy pointer translation, ctypes call overhead, and batch validation. Note on signed residuals: Because $t_{e2e}$ (100 iterations) and $t_{kernel}$ (500 iterations) are measured in separate benchmark sessions, workloads where pure kernel execution dominates latency (e.g., Model C at certain batch sizes) can exhibit small negative residuals (such as -1.90 μs) due to inter-session timer variance, core migration, or CPU frequency scaling; these are reported signed rather than clamped to zero.
5. ***Diagnostic Ratio (Column 11):** `PyTorch E2E / Tuned C++ Kernel`. This is NOT an equivalent-workload speedup because it divides PyTorch Python-level execution by native in-kernel execution. It serves as a diagnostic demonstrating the performance ceiling when framework interpreter overhead is completely eliminated.
