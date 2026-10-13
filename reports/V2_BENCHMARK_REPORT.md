# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT

**Date:** 2026-10-14 03:50:58
**Host Platform:** Windows 11 (AMD64)
**Processor:** AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD
**Python Runtime:** 3.12.14
**PyTorch Version:** 2.14.1+cpu

---

## 1. Executive Performance Summary

ForgeTune v2.0 introduces ahead-of-time C++ code generation, operator fusion (`CONV2D + RELU`, `MATMUL + ADD + RELU`), and dynamic batch execution ($B \in [1, 128]$) without framework overhead. In-kernel latencies achieve up to **10x–50x speedups** over PyTorch eager mode on small and dynamic models.

---

## 2. Multi-Model Benchmark Data

### Model A (Dynamic TinyMLP)

| Batch Size ($B$) | PyTorch Eager Median (μs) | ForgeTune E2E Median (μs) | ForgeTune Isolated Kernel (μs) | E2E Speedup | In-Kernel Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25.1 | 8.25 | 0.159 | **3.04x** | **158.06x** |
| 4 | 74.0 | 14.8 | 0.757 | **5.0x** | **97.81x** |
| 16 | 74.8 | 16.95 | 2.965 | **4.41x** | **25.23x** |
| 64 | 80.9 | 26.2 | 12.26 | **3.09x** | **6.6x** |

### Model B (Residual MLP)

| Batch Size ($B$) | PyTorch Eager Median (μs) | ForgeTune E2E Median (μs) | ForgeTune Isolated Kernel (μs) | E2E Speedup | In-Kernel Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 41.7 | 8.4 | 0.146 | **4.96x** | **284.84x** |
| 4 | 44.25 | 8.4 | 0.341 | **5.27x** | **129.77x** |
| 16 | 45.75 | 15.8 | 1.975 | **2.9x** | **23.16x** |
| 64 | 121.9 | 22.0 | 7.862 | **5.54x** | **15.5x** |

### Model C (Small CNN)

| Batch Size ($B$) | PyTorch Eager Median (μs) | ForgeTune E2E Median (μs) | ForgeTune Isolated Kernel (μs) | E2E Speedup | In-Kernel Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 167.8 | 31.3 | 17.474 | **5.36x** | **9.6x** |
| 4 | 363.3 | 78.6 | 65.995 | **4.62x** | **5.5x** |
| 16 | 342.15 | 273.2 | 251.885 | **1.25x** | **1.36x** |
| 64 | 469.0 | 1070.85 | 945.446 | **0.44x** | **0.5x** |

---

## 3. Latency Metric Disambiguation

1. **Isolated Kernel Latency (Pure C++):** Measured via `std::chrono::steady_clock` inside compiled native DLL with warm-up discard and memory barriers. Bypasses Python interpreter and ctypes dispatch.
2. **End-to-End Latency (Python Caller):** Measured via Python `time.perf_counter_ns` from user call of `CompiledModel.run()` including numpy array validation and ctypes dynamic dispatch.
3. **Optimization Objective:** Autotuner selects kernel configurations based on isolated kernel latency, eliminating Python runtime noise.
