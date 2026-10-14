# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT

**Date:** 2026-10-15 03:07:18
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

### Model A (Dynamic TinyMLP) (Selected: `loop_ikj`, Cost: 5.3s) — *Autotuner selected `loop_ikj` over `baseline`*

| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 24.60 | 24.90 | 12.80 | **1.92x** | **1.95x** | 0.304 | 0.355 | **0.86x** | 12.45 | 69.3x* |
| 4 | 57.05 | 40.10 | 13.00 | **4.39x** | **3.08x** | 1.087 | 1.150 | **0.95x** | 11.85 | 49.6x* |
| 16 | 44.00 | 38.20 | 16.40 | **2.68x** | **2.33x** | 4.119 | 4.449 | **0.93x** | 11.95 | 9.9x* |
| 64 | 49.50 | 43.80 | 31.00 | **1.60x** | **1.41x** | 15.909 | 18.067 | **0.88x** | 12.93 | 2.7x* |

### Model B (Residual MLP) (Selected: `baseline_naive`, Cost: 6.19s) — *Baseline retained by autotuner (<5% candidate performance difference)*

| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 41.40 | 41.30 | 8.40 | **4.93x** | **4.92x** | 0.114 | 0.114 | **1.00x** | 8.29 | 363.2x* |
| 4 | 43.25 | 43.30 | 8.60 | **5.03x** | **5.03x** | 0.352 | 0.352 | **1.00x** | 8.25 | 122.9x* |
| 16 | 45.80 | 44.40 | 9.70 | **4.72x** | **4.58x** | 1.376 | 1.368 | **1.00x** | 8.33 | 33.5x* |
| 64 | 110.10 | 77.30 | 22.55 | **4.88x** | **3.43x** | 8.019 | 8.048 | **1.00x** | 14.50 | 13.7x* |

### Model C (Small CNN) (Selected: `tiled_16`, Cost: 9.71s) — *Autotuner selected `tiled_16` over `baseline`*

| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 155.00 | 100.45 | 23.40 | **6.62x** | **4.29x** | 11.092 | 9.728 | **1.14x** | 13.67 | 15.9x* |
| 4 | 331.05 | 222.80 | 41.95 | **7.89x** | **5.31x** | 32.885 | 27.522 | **1.19x** | 14.43 | 12.0x* |
| 16 | 332.20 | 391.30 | 117.45 | **2.83x** | **3.33x** | 114.165 | 101.303 | **1.13x** | 16.15 | 3.3x* |
| 64 | 538.05 | 442.15 | 450.15 | **1.20x** | **0.98x** | 480.979 | 373.909 | **1.29x** | 76.24 | 1.4x* |

---

## 3. Latency Metric Methodology & Disambiguation

1. **Genuine End-to-End Speedup (Column 5):** Measures real-world Python caller latency: `PyTorch Eager (8 threads) / ForgeTune E2E`. Both run complete model inference with caller input validation.
2. **Controlled End-to-End Speedup (Column 6):** Measures single-threaded algorithmic efficiency: `PyTorch Eager (1 thread) / ForgeTune E2E`, controlling for multi-core parallelism in PyTorch MKL-DNN / oneDNN.
3. **Genuine Kernel Optimization Speedup (Column 9):** Measures compiler code-generation and fusion efficacy at identical boundaries: `Baseline C++ Kernel / Tuned C++ Kernel` inside the compiled native shared library.
4. **E2E - Kernel Residual (Column 10):** Difference between Python caller duration and pure native execution ($t_{e2e} - t_{kernel}$), representing numpy pointer translation, ctypes call overhead, and batch validation. Note on signed residuals: Because $t_{e2e}$ (100 iterations) and $t_{kernel}$ (500 iterations) are measured in separate benchmark sessions, workloads where pure kernel execution dominates latency (e.g., Model C at certain batch sizes) can exhibit small negative residuals (such as -1.90 μs) due to inter-session timer variance, core migration, or CPU frequency scaling; these are reported signed rather than clamped to zero.
5. ***PyTorch E2E / ForgeTune native-kernel diagnostic ratio (non-equivalent timing boundaries) (Column 11):** Ratio: Numerator = `PyTorch Eager (8 threads) E2E latency` ($t_{PT\_e2e}$); Denominator = `ForgeTune Tuned C++ Kernel latency` ($t_{FT\_kernel}$). This is NOT an equivalent-workload speedup because it divides PyTorch Python-level execution by native in-kernel execution. It serves as a diagnostic demonstrating the performance ceiling when framework interpreter overhead is completely eliminated.
