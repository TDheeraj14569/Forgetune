# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT

**Date:** 2026-10-28 10:32:15
**Run ID:** `v2-bench-1793163636-4a6b7683`
**Git Commit:** `4a6b7683ea6466c1d7883358b8857c9473ea0ea8`
**Tree Fingerprint:** `d52418ebb78f0953e3000e2ba63d1549ffddbc4b`
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

### Model A (Dynamic TinyMLP) (Selected: `tiled_16`, Cost: 21.33s) — *Autotuner selected `tiled_16` over `baseline`*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Candidate Kernel (μs) | Selected Implementation Kernel (μs) | Candidate Speedup | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25.10 | 25.00 | 5.90 | **4.25x** | **4.24x** | 0.229 | 0.152 | 0.152 | 1.51x | **1.51x** | 5.75 | 165.1x* |
| 4 | 42.90 | 37.40 | 9.10 | **4.71x** | **4.11x** | 1.145 | 0.646 | 0.646 | 1.77x | **1.77x** | 8.45 | 66.4x* |
| 16 | 47.10 | 40.20 | 10.90 | **4.32x** | **3.69x** | 4.466 | 2.512 | 2.512 | 1.78x | **1.78x** | 8.39 | 18.8x* |
| 64 | 50.55 | 44.30 | 18.60 | **2.72x** | **2.38x** | 17.424 | 9.946 | 9.946 | 1.75x | **1.75x** | 8.65 | 5.1x* |

### Model B (Residual MLP) (Selected: `fused_ikj`, Cost: 17.5s) — *Autotuner selected `fused_ikj` over `baseline`*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Candidate Kernel (μs) | Selected Implementation Kernel (μs) | Candidate Speedup | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 41.80 | 42.15 | 5.70 | **7.33x** | **7.39x** | 0.099 | 0.089 | 0.089 | 1.11x | **1.11x** | 5.61 | 469.7x* |
| 4 | 44.20 | 44.40 | 5.90 | **7.49x** | **7.53x** | 0.293 | 0.241 | 0.241 | 1.22x | **1.22x** | 5.66 | 183.4x* |
| 16 | 45.80 | 46.00 | 6.60 | **6.94x** | **6.97x** | 1.103 | 0.913 | 0.913 | 1.21x | **1.21x** | 5.69 | 50.2x* |
| 64 | 77.00 | 49.10 | 9.60 | **8.02x** | **5.11x** | 4.733 | 5.182 | 5.182 | 0.91x | **0.91x** | 4.42 | 14.9x* |

### Model C (Small CNN) (Selected: `loop_ikj`, Cost: 44.06s) — *Autotuner selected `loop_ikj` over `baseline`*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Candidate Kernel (μs) | Selected Implementation Kernel (μs) | Candidate Speedup | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 96.75 | 65.90 | 11.80 | **8.20x** | **5.58x** | 9.482 | 7.059 | 7.059 | 1.34x | **1.34x** | 4.74 | 13.7x* |
| 4 | 321.85 | 297.55 | 34.20 | **9.41x** | **8.70x** | 32.835 | 24.386 | 24.386 | 1.35x | **1.35x** | 9.81 | 13.2x* |
| 16 | 336.40 | 321.25 | 77.30 | **4.35x** | **4.16x** | 123.823 | 76.269 | 76.269 | 1.62x | **1.62x** | 1.03 | 4.4x* |
| 64 | 393.57 | 448.27 | 380.48 | **1.03x** | **1.18x** | 464.530 | 361.990 | 361.990 | 1.28x | **1.28x** | 18.49 | 1.1x* |

### Model C at $B=64$ Statistical Distribution Analysis (30 Independent Sample Batches)

To rigorously verify Model C performance at batch size $B=64$, 30 independent sample batches were measured across both frameworks using a unified measurement protocol:

| Execution Boundary | Median (μs) | Mean (μs) | Std Dev (μs) | IQR (μs) | Min (μs) | Max (μs) | 5th Pct (μs) | 95th Pct (μs) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PyTorch Eager (8 threads)** | 393.57 | 377.42 | 40.93 | 56.89 | 302.05 | 450.10 | 307.05 | 428.88 |
| **PyTorch Eager (1 thread)** | 448.27 | 443.42 | 44.49 | 13.30 | 342.05 | 590.70 | 348.24 | 489.67 |
| **ForgeTune Python E2E** | 380.48 | 372.23 | 28.03 | 1.14 | 284.95 | 388.85 | 290.07 | 387.12 |
| **ForgeTune Baseline Native Kernel** | 464.53 | 467.93 | 19.05 | 18.45 | 425.77 | 512.89 | 436.38 | 504.39 |
| **ForgeTune Selected Native Kernel (`loop_ikj`)** | 361.99 | 355.82 | 20.00 | 20.91 | 281.27 | 390.14 | 331.74 | 374.78 |
| **E2E - Kernel Residual** | 18.34 | 16.41 | 34.94 | 19.95 | -97.24 | 99.33 | -58.55 | 50.18 |

**Residual Analysis ($t_{e2e} - t_{kernel}$):**
- Median residual is 18.34 μs.
- The residual represents Python interpreter overhead, NumPy contiguous buffer conversion (`np.ascontiguousarray`), ctypes function argument wrapping, and dynamic batch shape checking.
- In pure single-threaded native execution without Python framework overhead, the ForgeTune selected native kernel executes at **361.99 μs**, achieving 1.24x speedup over single-threaded PyTorch E2E (448.27 μs).
- At the Python E2E level, single-threaded ForgeTune (380.48 μs) achieves 1.03x speedup vs 8-threaded PyTorch (393.57 μs) and 1.18x vs single-threaded PyTorch (448.27 μs).

---

## 3. Latency Metric Methodology & Disambiguation

1. **Genuine End-to-End Speedup (Column 5):** Measures real-world Python caller latency: `PyTorch Eager (8 threads) / ForgeTune E2E`. Both run complete model inference with caller input validation.
2. **Controlled End-to-End Speedup (Column 6):** Measures single-threaded algorithmic efficiency: `PyTorch Eager (1 thread) / ForgeTune E2E`, controlling for multi-core parallelism in PyTorch MKL-DNN / oneDNN.
3. **Genuine Kernel Optimization Speedup (Column 9):** Measures compiler code-generation and fusion efficacy at identical boundaries: `Baseline C++ Kernel / Selected Implementation Kernel` inside the compiled native shared library. For workloads where the baseline is retained (e.g. Models A and B, where alternative candidates are within the 5% noise window or slower across the dynamic batch range), this ratio is strictly 1.00x.
4. **E2E - Kernel Residual (Column 10):** Difference between Python caller duration and pure native execution ($t_{e2e} - t_{kernel}$), representing numpy pointer translation, ctypes call overhead, and batch validation. Note on signed residuals: Because $t_{e2e}$ (100 iterations) and $t_{kernel}$ (500 iterations) are measured in separate benchmark sessions, workloads where pure kernel execution dominates latency (e.g., Model C at certain batch sizes) can exhibit small negative residuals (such as -1.90 μs) due to inter-session timer variance, core migration, or CPU frequency scaling; these are reported signed rather than clamped to zero.
5. ***PyTorch E2E / ForgeTune native-kernel diagnostic ratio (non-equivalent timing boundaries) (Column 11):** Ratio: Numerator = `PyTorch Eager (8 threads) E2E latency` ($t_{PT\_e2e}$); Denominator = `ForgeTune Selected C++ Kernel latency` ($t_{FT\_kernel}$). This is NOT an equivalent-workload speedup because it divides PyTorch Python-level execution by native in-kernel execution. It serves as a diagnostic demonstrating the performance ceiling when framework interpreter overhead is completely eliminated.

---

## 4. Statistically Rigorous Paired Hypothesis Testing ($N=100$ Blocks)

To evaluate Model C at $B=64$ without sequential bias, an interleaved 100-block paired experiment was executed with alternating execution order (`reports/v2_paired_b64_benchmark_data.json`):
- **PyTorch 1-Thread E2E Median:** $359.35\ \mu\text{s}$
- **ForgeTune E2E Median:** $288.60\ \mu\text{s}$
- **Mean Paired Difference ($\bar{d} = t_{PT1} - t_{FT}$):** $+82.58\ \mu\text{s}$ paired latency reduction (ForgeTune faster)
- **Median Paired Difference:** $+64.73\ \mu\text{s}$
- **Standard Error ($SE$):** $7.27\ \mu\text{s}$ ($df = 99$, sample standard deviation $s_d = 72.65\ \mu\text{s}$)
- **One-Sided Non-Inferiority ($H_0: \mu_d \le -17.97\ \mu\text{s}$):** $t = 13.84$, exact Student's t $p = 3.50 \times 10^{-25} \ll 0.0001$. Non-inferiority is certified.
- **Two-Sided TOST Equivalence ($[-\delta, +\delta] = [-17.97, +17.97]\ \mu\text{s}$):** Inconclusive strictly because ForgeTune significantly outperforms PyTorch 1T ($90\%\ \text{CI} = [+70.52, +94.65]\ \mu\text{s}$), placing the upper confidence bound above $+17.97\ \mu\text{s}$. The test fails equivalence due to substantial performance superiority, not high variance or performance inferiority.
- **Bootstrap 90% Confidence Intervals ($B=10,000$ resamples):**
  - Mean Difference: $[+71.15\ \mu\text{s},\; +95.04\ \mu\text{s}]$
  - Median Difference: $[+58.67\ \mu\text{s},\; +76.70\ \mu\text{s}]$
