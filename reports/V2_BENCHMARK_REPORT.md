# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT

**Date:** 2026-10-14 14:45:19
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

### Model A (Dynamic TinyMLP) (Compilation Cost: 36.89s)

| Batch Size ($B$) | PyTorch Eager (8th) (μs) | PyTorch Eager (1th) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8th)** | **Controlled E2E Speedup (1th)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25.30 | 25.30 | 8.70 | **2.91x** | **2.91x** | 0.228 | 0.229 | **1.00x** | 8.47 | 110.5x* |
| 4 | 43.55 | 37.90 | 13.50 | **3.23x** | **2.81x** | 0.923 | 1.163 | **0.79x** | 12.34 | 37.4x* |
| 16 | 44.40 | 39.40 | 16.50 | **2.69x** | **2.39x** | 4.177 | 4.444 | **0.94x** | 12.06 | 10.0x* |
| 64 | 49.30 | 45.20 | 30.50 | **1.62x** | **1.48x** | 18.003 | 17.664 | **1.02x** | 12.84 | 2.8x* |

### Model B (Residual MLP) (Compilation Cost: 35.91s)

| Batch Size ($B$) | PyTorch Eager (8th) (μs) | PyTorch Eager (1th) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8th)** | **Controlled E2E Speedup (1th)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 42.20 | 42.45 | 8.40 | **5.02x** | **5.05x** | 0.117 | 0.117 | **1.00x** | 8.28 | 360.7x* |
| 4 | 44.70 | 44.70 | 8.70 | **5.14x** | **5.14x** | 0.375 | 0.394 | **0.95x** | 8.31 | 113.5x* |
| 16 | 46.10 | 47.90 | 9.70 | **4.75x** | **4.94x** | 1.453 | 1.455 | **1.00x** | 8.24 | 31.7x* |
| 64 | 79.70 | 69.60 | 20.70 | **3.85x** | **3.36x** | 7.963 | 7.966 | **1.00x** | 12.73 | 10.0x* |

### Model C (Small CNN) (Compilation Cost: 39.56s)

| Batch Size ($B$) | PyTorch Eager (8th) (μs) | PyTorch Eager (1th) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8th)** | **Controlled E2E Speedup (1th)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 71.40 | 66.05 | 22.80 | **3.13x** | **2.90x** | 18.941 | 14.121 | **1.34x** | 8.68 | 5.1x* |
| 4 | 368.40 | 310.85 | 80.90 | **4.55x** | **3.84x** | 74.352 | 67.960 | **1.09x** | 12.94 | 5.4x* |
| 16 | 351.10 | 320.95 | 284.90 | **1.23x** | **1.13x** | 270.492 | 266.102 | **1.02x** | 18.80 | 1.3x* |
| 64 | 415.75 | 442.80 | 1103.65 | **0.38x** | **0.40x** | 1053.998 | 935.310 | **1.13x** | 168.34 | 0.4x* |

---

## 3. Latency Metric Methodology & Disambiguation

1. **Genuine End-to-End Speedup (Column 5):** Measures real-world Python caller latency: `PyTorch Eager (Default 8th) / ForgeTune E2E`. Both run complete model inference with caller input validation.
2. **Controlled End-to-End Speedup (Column 6):** Measures single-threaded algorithmic efficiency: `PyTorch Eager (1th) / ForgeTune E2E`, controlling for multi-core parallelism in PyTorch MKL-DNN / oneDNN.
3. **Genuine Kernel Optimization Speedup (Column 9):** Measures compiler code-generation and fusion efficacy at identical boundaries: `Baseline C++ Kernel / Tuned C++ Kernel` inside the compiled native shared library.
4. **E2E - Kernel Residual (Column 10):** Difference between Python caller duration and pure native execution ($t_{e2e} - t_{kernel}$), representing numpy pointer translation, ctypes call overhead, and batch validation. Note on signed residuals: Because $t_{e2e}$ (100 iterations) and $t_{kernel}$ (500 iterations) are measured in separate benchmark sessions, workloads where pure kernel execution dominates latency (e.g., Model C at certain batch sizes) can exhibit small negative residuals (such as -1.90 μs) due to inter-session timer variance, core migration, or CPU frequency scaling; these are reported signed rather than clamped to zero.
5. ***Diagnostic Ratio (Column 11):** `PyTorch E2E / Tuned C++ Kernel`. This is NOT an equivalent-workload speedup because it divides PyTorch Python-level execution by native in-kernel execution. It serves as a diagnostic demonstrating the performance ceiling when framework interpreter overhead is completely eliminated.
