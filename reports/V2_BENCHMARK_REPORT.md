# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT

**Date:** 2026-10-14 05:03:15
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

### Model A (Dynamic TinyMLP) (Compilation Cost: 3.24s)

| Batch Size ($B$) | PyTorch Eager (8th) (μs) | PyTorch Eager (1th) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8th)** | **Controlled E2E Speedup (1th)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25.45 | 25.75 | 12.70 | **2.00x** | **2.03x** | 0.281 | 0.304 | **0.92x** | 12.40 | 83.7x* |
| 4 | 44.10 | 38.60 | 9.70 | **4.55x** | **3.98x** | 1.027 | 1.101 | **0.93x** | 8.60 | 40.1x* |
| 16 | 35.30 | 28.40 | 11.85 | **2.98x** | **2.40x** | 3.196 | 3.609 | **0.89x** | 8.24 | 9.8x* |
| 64 | 55.00 | 50.10 | 30.00 | **1.83x** | **1.67x** | 17.885 | 17.097 | **1.05x** | 12.90 | 3.2x* |

### Model B (Residual MLP) (Compilation Cost: 6.22s)

| Batch Size ($B$) | PyTorch Eager (8th) (μs) | PyTorch Eager (1th) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8th)** | **Controlled E2E Speedup (1th)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 67.25 | 68.15 | 13.90 | **4.84x** | **4.90x** | 0.152 | 0.154 | **0.99x** | 13.75 | 436.7x* |
| 4 | 70.25 | 70.60 | 13.70 | **5.13x** | **5.15x** | 0.664 | 0.507 | **1.31x** | 13.19 | 138.6x* |
| 16 | 72.50 | 48.15 | 15.70 | **4.62x** | **3.07x** | 1.466 | 1.979 | **0.74x** | 13.72 | 36.6x* |
| 64 | 77.90 | 77.10 | 22.40 | **3.48x** | **3.44x** | 8.164 | 8.204 | **1.00x** | 14.20 | 9.5x* |

### Model C (Small CNN) (Compilation Cost: 14.42s)

| Batch Size ($B$) | PyTorch Eager (8th) (μs) | PyTorch Eager (1th) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8th)** | **Controlled E2E Speedup (1th)** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 145.10 | 103.35 | 33.40 | **4.34x** | **3.09x** | 33.318 | 18.590 | **1.79x** | 14.81 | 7.8x* |
| 4 | 385.40 | 230.55 | 65.60 | **5.88x** | **3.51x** | 71.214 | 67.496 | **1.06x** | 0.00 | 5.7x* |
| 16 | 344.20 | 350.00 | 288.40 | **1.19x** | **1.21x** | 258.786 | 253.753 | **1.02x** | 34.65 | 1.4x* |
| 64 | 426.05 | 488.35 | 1118.00 | **0.38x** | **0.44x** | 1121.199 | 978.263 | **1.15x** | 139.74 | 0.4x* |

---

## 3. Latency Metric Methodology & Disambiguation

1. **Genuine End-to-End Speedup (Column 5):** Measures real-world Python caller latency: `PyTorch Eager (Default 8th) / ForgeTune E2E`. Both run complete model inference with caller input validation.
2. **Controlled End-to-End Speedup (Column 6):** Measures single-threaded algorithmic efficiency: `PyTorch Eager (1th) / ForgeTune E2E`, controlling for multi-core parallelism in PyTorch MKL-DNN / oneDNN.
3. **Genuine Kernel Optimization Speedup (Column 9):** Measures compiler code-generation and fusion efficacy at identical boundaries: `Baseline C++ Kernel / Tuned C++ Kernel` inside the compiled native shared library.
4. **E2E - Kernel Residual (Column 10):** Difference between Python caller duration and pure native execution ($t_{e2e} - t_{kernel}$), representing numpy pointer translation, ctypes call overhead, and batch validation.
5. ***Diagnostic Ratio (Column 11):** `PyTorch E2E / Tuned C++ Kernel`. This is NOT an equivalent-workload speedup because it divides PyTorch Python-level execution by native in-kernel execution. It serves as a diagnostic demonstrating the performance ceiling when framework interpreter overhead is completely eliminated.
