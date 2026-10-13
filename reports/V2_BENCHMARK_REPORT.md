# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT

**Date:** 2026-10-14 04:38:30
**Host Platform:** Windows 11 (AMD64)
**Processor:** AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD
**Python Runtime:** 3.12.14
**PyTorch Version:** 2.14.1+cpu

---

## 1. Executive Performance Summary

ForgeTune v2.0 introduces ahead-of-time C++ code generation, operator fusion (`CONV2D + RELU`, `MATMUL + ADD + RELU`), and dynamic batch execution ($B \in [1, 128]$) without framework overhead. Benchmarks distinguish:
1. **Genuine End-to-End Speedup:** Equivalent boundaries comparing PyTorch eager execution in Python vs ForgeTune execution in Python.
2. **Genuine Native Kernel Optimization Speedup:** Equivalent boundaries comparing ForgeTune unoptimized native C++ baseline vs ForgeTune autotuned fused native C++ kernel.
3. **Dispatch Overhead:** Python ctypes dynamic dispatch and buffer marshaling overhead ($t_{e2e} - t_{kernel}$).
4. **Diagnostic Boundary Ratio:** Framework overhead elimination metric ($t_{PT\_e2e} / t_{FT\_kernel}$).

---

## 2. Multi-Model Benchmark Data

### Model A (Dynamic TinyMLP)

| Batch Size ($B$) | PyTorch Eager E2E (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | Dispatch Overhead (μs) | Diagnostic E2E-to-Kernel Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 24.8 | 14.3 | **1.73x** | 0.305 | 0.282 | **1.08x** | 14.02 | 87.94x* |
| 4 | 88.2 | 14.6 | **6.04x** | 1.022 | 4.991 | **0.2x** | 9.61 | 17.67x* |
| 16 | 80.65 | 19.45 | **4.15x** | 4.208 | 4.643 | **0.91x** | 14.81 | 17.37x* |
| 64 | 72.8 | 28.35 | **2.57x** | 16.368 | 16.23 | **1.01x** | 12.12 | 4.49x* |

### Model B (Residual MLP)

| Batch Size ($B$) | PyTorch Eager E2E (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | Dispatch Overhead (μs) | Diagnostic E2E-to-Kernel Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 41.9 | 8.4 | **4.99x** | 0.116 | 0.115 | **1.01x** | 8.28 | 363.72x* |
| 4 | 44.3 | 8.3 | **5.34x** | 0.357 | 0.353 | **1.01x** | 7.95 | 125.42x* |
| 16 | 45.7 | 9.7 | **4.71x** | 1.421 | 1.386 | **1.03x** | 8.31 | 32.97x* |
| 64 | 109.8 | 22.1 | **4.97x** | 8.647 | 8.244 | **1.05x** | 13.86 | 13.32x* |

### Model C (Small CNN)

| Batch Size ($B$) | PyTorch Eager E2E (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup** | Baseline C++ Kernel (μs) | Tuned C++ Kernel (μs) | **Genuine Kernel Opt Speedup** | Dispatch Overhead (μs) | Diagnostic E2E-to-Kernel Ratio* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 154.7 | 31.6 | **4.9x** | 18.472 | 16.966 | **1.09x** | 14.63 | 9.12x* |
| 4 | 350.05 | 65.2 | **5.37x** | 71.522 | 65.35 | **1.09x** | 0.0 | 5.36x* |
| 16 | 335.0 | 232.75 | **1.44x** | 269.903 | 257.537 | **1.05x** | 0.0 | 1.3x* |
| 64 | 393.05 | 1108.55 | **0.35x** | 1107.14 | 1014.87 | **1.09x** | 93.68 | 0.39x* |

---

## 3. Latency Metric Methodology & Disambiguation

1. **Genuine End-to-End Speedup (Column 4):** Measures real-world Python caller latency: `PyTorch Eager E2E / ForgeTune E2E`. Both run complete model inference with caller input validation.
2. **Genuine Kernel Optimization Speedup (Column 7):** Measures compiler code-generation and fusion efficacy at identical boundaries: `Baseline C++ Kernel / Tuned C++ Kernel` inside the compiled native shared library.
3. **Dispatch Overhead (Column 8):** Difference between Python caller duration and pure native execution ($t_{e2e} - t_{kernel}$), representing numpy pointer translation and ctypes call overhead.
4. ***Diagnostic Ratio (Column 9):** `PyTorch E2E / Tuned C++ Kernel`. This is NOT an equivalent-workload speedup because it divides PyTorch Python-level execution by native in-kernel execution. It serves as a diagnostic demonstrating the performance ceiling when framework interpreter overhead is completely eliminated.
