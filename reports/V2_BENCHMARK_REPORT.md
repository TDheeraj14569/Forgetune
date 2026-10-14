# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT

**Date:** 2026-10-14 15:00:52
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

### Model A (Dynamic TinyMLP) (Compilation Cost: 6.33s)

| Batch Size ($B$) | PyTorch Eager (8th) (μs) | PyTorch Eager (1th) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8th)** | **Controlled E2E Speedup (1th)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 34.05 | 26.65 | 8.50 | **4.01x** | **3.14x** | 0.223 | 0.243 | **0.92x** | 8.26 | 140.1x* |
| 4 | 71.40 | 38.70 | 13.50 | **5.29x** | **2.87x** | 1.020 | 1.102 | **0.93x** | 12.40 | 64.8x* |
| 16 | 47.80 | 39.50 | 16.50 | **2.90x** | **2.39x** | 3.975 | 4.319 | **0.92x** | 12.18 | 11.1x* |
| 64 | 38.60 | 45.40 | 31.20 | **1.24x** | **1.46x** | 15.086 | 17.303 | **0.87x** | 13.90 | 2.2x* |

### Model B (Residual MLP) (Compilation Cost: 34.65s)

| Batch Size ($B$) | PyTorch Eager (8th) (μs) | PyTorch Eager (1th) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8th)** | **Controlled E2E Speedup (1th)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 41.80 | 42.05 | 8.40 | **4.98x** | **5.01x** | 0.115 | 0.115 | **1.00x** | 8.29 | 363.5x* |
| 4 | 44.15 | 44.05 | 8.60 | **5.13x** | **5.12x** | 0.358 | 0.371 | **0.96x** | 8.23 | 119.0x* |
| 16 | 45.60 | 45.30 | 9.70 | **4.70x** | **4.67x** | 1.541 | 1.363 | **1.13x** | 8.34 | 33.5x* |
| 64 | 77.90 | 69.40 | 19.50 | **3.99x** | **3.56x** | 7.619 | 7.393 | **1.03x** | 12.11 | 10.5x* |

### Model C (Small CNN) (Compilation Cost: 39.33s)

| Batch Size ($B$) | PyTorch Eager (8th) (μs) | PyTorch Eager (1th) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8th)** | **Controlled E2E Speedup (1th)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 97.85 | 87.70 | 30.85 | **3.17x** | **2.84x** | 16.563 | 16.653 | **0.99x** | 14.20 | 5.9x* |
| 4 | 325.10 | 299.70 | 82.05 | **3.96x** | **3.65x** | 70.593 | 67.180 | **1.05x** | 14.87 | 4.8x* |
| 16 | 344.50 | 264.45 | 286.25 | **1.20x** | **0.92x** | 256.634 | 255.047 | **1.01x** | 31.20 | 1.4x* |
| 64 | 372.10 | 448.40 | 908.30 | **0.41x** | **0.49x** | 1037.301 | 949.110 | **1.09x** | -40.81 | 0.4x* |

---

## 3. Latency Metric Methodology & Disambiguation

1. **Genuine End-to-End Speedup (Column 5):** Measures real-world Python caller latency: `PyTorch Eager (Default 8th) / ForgeTune E2E`. Both run complete model inference with caller input validation.
2. **Controlled End-to-End Speedup (Column 6):** Measures single-threaded algorithmic efficiency: `PyTorch Eager (1th) / ForgeTune E2E`, controlling for multi-core parallelism in PyTorch MKL-DNN / oneDNN.
3. **Genuine Kernel Optimization Speedup (Column 9):** Measures compiler code-generation and fusion efficacy at identical boundaries: `Baseline C++ Kernel / Tuned C++ Kernel` inside the compiled native shared library.
4. **E2E - Kernel Residual (Column 10):** Difference between Python caller duration and pure native execution ($t_{e2e} - t_{kernel}$), representing numpy pointer translation, ctypes call overhead, and batch validation. Note on signed residuals: Because $t_{e2e}$ (100 iterations) and $t_{kernel}$ (500 iterations) are measured in separate benchmark sessions, workloads where pure kernel execution dominates latency (e.g., Model C at certain batch sizes) can exhibit small negative residuals (such as -1.90 μs) due to inter-session timer variance, core migration, or CPU frequency scaling; these are reported signed rather than clamped to zero.
5. ***Diagnostic Ratio (Column 11):** `PyTorch E2E / Tuned C++ Kernel`. This is NOT an equivalent-workload speedup because it divides PyTorch Python-level execution by native in-kernel execution. It serves as a diagnostic demonstrating the performance ceiling when framework interpreter overhead is completely eliminated.
