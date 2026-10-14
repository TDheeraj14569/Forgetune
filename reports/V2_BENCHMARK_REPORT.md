# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT

**Date:** 2026-10-15 02:57:33
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

### Model A (Dynamic TinyMLP) (Selected: `baseline_naive`, Cost: 5.44s) — *Baseline retained by autotuner (<5% candidate performance difference)*

| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 24.80 | 24.80 | 8.20 | **3.02x** | **3.02x** | 0.222 | 0.240 | **1.00x** | 7.96 | 103.3x* |
| 4 | 60.35 | 40.20 | 14.40 | **4.19x** | **2.79x** | 1.122 | 1.132 | **1.00x** | 13.27 | 53.3x* |
| 16 | 53.10 | 39.00 | 16.30 | **3.26x** | **2.39x** | 4.017 | 4.001 | **1.00x** | 12.30 | 13.3x* |
| 64 | 49.50 | 44.20 | 28.75 | **1.72x** | **1.54x** | 16.133 | 16.094 | **1.00x** | 12.66 | 3.1x* |

### Model B (Residual MLP) (Selected: `baseline_naive`, Cost: 23.59s) — *Baseline retained by autotuner (<5% candidate performance difference)*

| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 57.65 | 41.30 | 13.40 | **4.30x** | **3.08x** | 0.162 | 0.163 | **1.00x** | 13.24 | 353.7x* |
| 4 | 67.20 | 43.10 | 8.70 | **7.72x** | **4.95x** | 0.358 | 0.354 | **1.00x** | 8.35 | 189.8x* |
| 16 | 44.80 | 44.50 | 13.80 | **3.25x** | **3.22x** | 1.375 | 1.362 | **1.00x** | 12.44 | 32.9x* |
| 64 | 156.40 | 76.80 | 22.30 | **7.01x** | **3.44x** | 8.451 | 8.452 | **1.00x** | 13.85 | 18.5x* |

### Model C (Small CNN) (Selected: `tiled_16`, Cost: 27.86s) — *Autotuner selected `tiled_16` over `baseline`*

| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 181.50 | 100.40 | 23.50 | **7.72x** | **4.27x** | 9.642 | 8.617 | **1.12x** | 14.88 | 21.1x* |
| 4 | 352.95 | 303.65 | 41.40 | **8.53x** | **7.33x** | 29.092 | 26.911 | **1.08x** | 14.49 | 13.1x* |
| 16 | 324.25 | 374.35 | 124.10 | **2.61x** | **3.02x** | 114.995 | 98.371 | **1.17x** | 25.73 | 3.3x* |
| 64 | 493.15 | 448.60 | 475.70 | **1.04x** | **0.94x** | 478.132 | 368.540 | **1.30x** | 107.16 | 1.3x* |

---

## 3. Latency Metric Methodology & Disambiguation

1. **Genuine End-to-End Speedup (Column 5):** Measures real-world Python caller latency: `PyTorch Eager (8 threads) / ForgeTune E2E`. Both run complete model inference with caller input validation.
2. **Controlled End-to-End Speedup (Column 6):** Measures single-threaded algorithmic efficiency: `PyTorch Eager (1 thread) / ForgeTune E2E`, controlling for multi-core parallelism in PyTorch MKL-DNN / oneDNN.
3. **Genuine Kernel Optimization Speedup (Column 9):** Measures compiler code-generation and fusion efficacy at identical boundaries: `Baseline C++ Kernel / Tuned C++ Kernel` inside the compiled native shared library.
4. **E2E - Kernel Residual (Column 10):** Difference between Python caller duration and pure native execution ($t_{e2e} - t_{kernel}$), representing numpy pointer translation, ctypes call overhead, and batch validation. Note on signed residuals: Because $t_{e2e}$ (100 iterations) and $t_{kernel}$ (500 iterations) are measured in separate benchmark sessions, workloads where pure kernel execution dominates latency (e.g., Model C at certain batch sizes) can exhibit small negative residuals (such as -1.90 μs) due to inter-session timer variance, core migration, or CPU frequency scaling; these are reported signed rather than clamped to zero.
5. ***PyTorch E2E / ForgeTune native-kernel diagnostic ratio (non-equivalent timing boundaries) (Column 11):** Ratio: Numerator = `PyTorch Eager (8 threads) E2E latency` ($t_{PT\_e2e}$); Denominator = `ForgeTune Tuned C++ Kernel latency` ($t_{FT\_kernel}$). This is NOT an equivalent-workload speedup because it divides PyTorch Python-level execution by native in-kernel execution. It serves as a diagnostic demonstrating the performance ceiling when framework interpreter overhead is completely eliminated.
