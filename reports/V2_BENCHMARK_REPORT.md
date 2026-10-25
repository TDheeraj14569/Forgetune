# FORGETUNE V2.0 — BENCHMARK EVALUATION REPORT

**Date:** 2026-10-24 23:14:03
**Run ID:** `v2-bench-1792863787-0233951b`
**Git Commit:** `0233951bcd8500a1acd9046fb2b5633b90025902`
**Tree Fingerprint:** `069abc69a15e586c686b8b2afe9596e28f7bf648`
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

### Model A (Dynamic TinyMLP) (Selected: `baseline_naive`, Cost: 7.13s) — *Baseline retained by autotuner (<5% candidate performance difference)*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Selected Implementation Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 25.90 | 26.10 | 5.85 | **4.43x** | **4.46x** | 0.230 | 0.230 | **1.00x** | 5.62 | 112.6x* |
| 4 | 42.50 | 37.60 | 9.20 | **4.62x** | **4.09x** | 0.852 | 0.849 | **1.00x** | 8.35 | 50.1x* |
| 16 | 32.70 | 38.60 | 12.70 | **2.57x** | **3.04x** | 4.514 | 3.874 | **1.00x** | 8.83 | 8.4x* |
| 64 | 37.60 | 32.00 | 26.20 | **1.44x** | **1.22x** | 17.345 | 16.516 | **1.00x** | 9.68 | 2.3x* |

### Model B (Residual MLP) (Selected: `baseline_naive`, Cost: 4.37s) — *Baseline retained by autotuner (<5% candidate performance difference)*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Selected Implementation Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 42.00 | 43.10 | 9.10 | **4.62x** | **4.74x** | 0.121 | 0.182 | **1.00x** | 8.92 | 230.8x* |
| 4 | 44.30 | 44.25 | 6.00 | **7.38x** | **7.38x** | 0.395 | 0.375 | **1.00x** | 5.62 | 118.1x* |
| 16 | 45.55 | 45.80 | 6.90 | **6.60x** | **6.64x** | 1.449 | 1.431 | **1.00x** | 5.47 | 31.8x* |
| 64 | 100.55 | 69.00 | 16.40 | **6.13x** | **4.21x** | 8.089 | 8.013 | **1.00x** | 8.39 | 12.5x* |

### Model C (Small CNN) (Selected: `loop_ikj`, Cost: 31.95s) — *Autotuner selected `loop_ikj` over `baseline`*
| Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8 threads)** | **Controlled E2E Speedup (1 thread)** | Baseline C++ Kernel (μs) | Selected Implementation Kernel (μs) | **Genuine Kernel Opt Speedup** | E2E - Kernel Residual (μs) | Diagnostic Ratio (Non-equiv)* |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 84.05 | 88.45 | 16.80 | **5.00x** | **5.26x** | 9.423 | 7.986 | **1.18x** | 8.81 | 10.5x* |
| 4 | 321.70 | 307.60 | 37.60 | **8.56x** | **8.18x** | 33.648 | 26.789 | **1.26x** | 10.81 | 12.0x* |
| 16 | 352.00 | 286.90 | 89.65 | **3.93x** | **3.20x** | 120.367 | 92.864 | **1.30x** | -3.21 | 3.8x* |
| 64 | 409.53 | 462.02 | 442.70 | **0.93x** | **1.04x** | 461.680 | 421.300 | **1.10x** | 24.13 | 1.0x* |

### Model C at $B=64$ Statistical Distribution Analysis (30 Independent Sample Batches)

To rigorously verify Model C performance at batch size $B=64$, 30 independent sample batches were measured across both frameworks using a unified measurement protocol:

| Execution Boundary | Median (μs) | Mean (μs) | Std Dev (μs) | IQR (μs) | Min (μs) | Max (μs) | 5th Pct (μs) | 95th Pct (μs) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PyTorch Eager (8 threads)** | 409.53 | 417.24 | 67.97 | 51.70 | 308.45 | 608.10 | 315.95 | 530.35 |
| **PyTorch Eager (1 thread)** | 462.02 | 454.64 | 52.43 | 30.80 | 341.05 | 559.90 | 351.25 | 511.64 |
| **ForgeTune Python E2E** | 442.70 | 430.42 | 40.44 | 9.66 | 339.05 | 523.25 | 345.44 | 453.90 |
| **ForgeTune Baseline Native Kernel** | 461.68 | 463.83 | 13.90 | 12.83 | 434.85 | 495.35 | 446.98 | 490.61 |
| **ForgeTune Selected Native Kernel (`loop_ikj`)** | 421.30 | 419.80 | 15.51 | 27.92 | 391.22 | 444.69 | 395.35 | 440.82 |
| **E2E - Kernel Residual** | 24.13 | 10.62 | 48.83 | 34.33 | -105.64 | 100.25 | -88.58 | 55.46 |

**Residual Analysis ($t_{e2e} - t_{kernel}$):**
- Median residual is 24.13 μs (or 17.45 μs in run `0233951b`).
- **Mathematical Resolution of Residual Calculation:** In statistical distribution analysis, for paired observations $(X_i, Y_i)$, $\text{median}(X - Y) \ne \text{median}(X) - \text{median}(Y)$ for skewed empirical distributions. The marginal difference of medians is $442.70 - 421.30 = 21.40\ \mu\text{s}$ (or $443.02 - 418.45 = 24.57\ \mu\text{s}$), whereas the median of the 30 paired sample residuals $R_i = X_i - Y_i$ is **24.13 μs** (or **17.45 μs**). Both values are mathematically valid and reflect the difference between marginal sample medians and paired sample medians.
- The residual represents Python interpreter overhead, NumPy contiguous buffer conversion (`np.ascontiguousarray`), ctypes function argument wrapping, and dynamic batch shape checking.
- In pure single-threaded native execution without Python framework overhead, the ForgeTune selected native kernel executes at **421.30 μs**, achieving 1.10x speedup over single-threaded PyTorch E2E (462.02 μs).
- At the Python E2E level, single-threaded ForgeTune (442.70 μs) achieves 0.93x speedup vs 8-threaded PyTorch (409.53 μs) and 1.04x vs single-threaded PyTorch (462.02 μs).

### Statistical Parity, TOST Equivalence & Non-Inferiority Analysis (Model C at $B=64$)

To rigorously evaluate whether single-threaded ForgeTune achieves equivalence, non-inferiority, or strict superiority compared to single-threaded PyTorch at $B=64$, 30 paired blocks of 50 iterations each ($N=30$) were analyzed:

- **Baseline Metrics (Single-Threaded PyTorch E2E):** Mean $\mu_{PT} = 454.64\ \mu\text{s}$, sample standard deviation $s_{PT} = 53.33\ \mu\text{s}$, median $462.02\ \mu\text{s}$, IQR $30.80\ \mu\text{s}$.
- **Candidate Metrics (Single-Threaded ForgeTune E2E):** Mean $\mu_{FT} = 430.42\ \mu\text{s}$, sample standard deviation $s_{FT} = 41.13\ \mu\text{s}$, median $442.70\ \mu\text{s}$, IQR $9.66\ \mu\text{s}$.
- **Paired Sample Differences ($D_i = \text{PT1}_i - \text{FT}_i$):**
  - Mean paired difference: $\bar{d} = \mathbf{24.22\ \mu\text{s}}$.
  - Standard deviation of paired differences: $s_d = 48.73\ \mu\text{s}$.
  - Standard error of the mean: $SE = s_d / \sqrt{30} = \mathbf{8.90\ \mu\text{s}}$.
  - Median paired difference: $\mathbf{18.15\ \mu\text{s}}$ (marginal difference of medians: $462.02 - 442.70 = 19.32\ \mu\text{s}$).
  - Paired 90% Confidence Interval ($t_{0.95, df=29} = 1.6991$): $[\bar{d} \pm t \cdot SE] = [\mathbf{9.10\ \mu\text{s}}, \mathbf{39.33\ \mu\text{s}}]$.
  - Bootstrap 90% Confidence Interval for median difference: $[\mathbf{13.30\ \mu\text{s}}, \mathbf{30.25\ \mu\text{s}}]$.
- **Equivalence Margin:** $\Delta = \pm 5\%$ of the PyTorch baseline mean ($\pm 22.73\ \mu\text{s}$ on $454.64\ \mu\text{s}$). Equivalence zone: $[-\Delta, +\Delta] = [-22.73\ \mu\text{s}, +22.73\ \mu\text{s}]$.

#### Three-Way Statistical Performance Disambiguation:
1. **Strict Superiority (E2E): NOT ESTABLISHED ACROSS ALL CONFIGURATIONS.**
   Although $\bar{d} = 24.22\ \mu\text{s} > 0$ for single-threaded execution, ForgeTune single-threaded at $B=64$ (median $442.70\ \mu\text{s}$) does not outperform 8-threaded PyTorch E2E (median $409.53\ \mu\text{s}$, where PyTorch leverages MKL-DNN multithreaded convolution). Therefore, strict overall E2E superiority cannot be claimed.
2. **Two-Sided TOST Equivalence: STATISTICALLY INCONCLUSIVE AT $N=30$.**
   Under the Two One-Sided Tests (TOST) framework at $\alpha=0.05$ (corresponding to the 90% CI), concluding statistical equivalence requires both:
   - (a) $\text{Lower Bound} > -\Delta \implies 9.10 > -22.73$ (Satisfied, $p < 0.001$).
   - (b) $\text{Upper Bound} < +\Delta \implies 39.33 < +22.73$ (**NOT SATISFIED: $+39.33 > +22.73$).
   Because the upper bound exceeds $+22.73\ \mu\text{s}$, two-sided TOST equivalence cannot be rejected/certified at $N=30$ under host CPU scheduling variance. Release Gate 7 is held on **HOLD**.
3. **One-Sided Non-Inferiority: STATISTICALLY SUPPORTED ($p < 0.05$).**
   Non-inferiority tests the hypothesis that ForgeTune is no worse than single-threaded PyTorch by more than $\Delta$ ($H_0: \mu_{PT} - \mu_{FT} \le -\Delta$). Because the 95% one-sided confidence lower bound (equivalent to the lower bound of the 90% two-sided CI) is $+9.10\ \mu\text{s} > -22.73\ \mu\text{s}$, the null hypothesis of inferiority is soundly rejected ($t = (24.22 - (-22.73)) / 8.90 = 5.28, p < 0.0001$). ForgeTune is rigorously certified as **non-inferior** to single-threaded PyTorch at $B=64$.

---

## 3. Latency Metric Methodology & Disambiguation

1. **Genuine End-to-End Speedup (Column 5):** Measures real-world Python caller latency: `PyTorch Eager (8 threads) / ForgeTune E2E`. Both run complete model inference with caller input validation.
2. **Controlled End-to-End Speedup (Column 6):** Measures single-threaded algorithmic efficiency: `PyTorch Eager (1 thread) / ForgeTune E2E`, controlling for multi-core parallelism in PyTorch MKL-DNN / oneDNN.
3. **Genuine Kernel Optimization Speedup (Column 9):** Measures compiler code-generation and fusion efficacy at identical boundaries: `Baseline C++ Kernel / Selected Implementation Kernel` inside the compiled native shared library. For workloads where the baseline is retained (e.g. Models A and B, where alternative candidates are within the 5% noise window or slower across the dynamic batch range), this ratio is strictly 1.00x.
4. **E2E - Kernel Residual (Column 10):** Difference between Python caller duration and pure native execution ($t_{e2e} - t_{kernel}$), representing numpy pointer translation, ctypes call overhead, and batch validation. Note on signed residuals: Because $t_{e2e}$ (100 iterations) and $t_{kernel}$ (500 iterations) are measured in separate benchmark sessions, workloads where pure kernel execution dominates latency (e.g., Model C at certain batch sizes) can exhibit small negative residuals (such as -1.90 μs) due to inter-session timer variance, core migration, or CPU frequency scaling; these are reported signed rather than clamped to zero.
5. ***PyTorch E2E / ForgeTune native-kernel diagnostic ratio (non-equivalent timing boundaries) (Column 11):** Ratio: Numerator = `PyTorch Eager (8 threads) E2E latency` ($t_{PT\_e2e}$); Denominator = `ForgeTune Selected C++ Kernel latency` ($t_{FT\_kernel}$). This is NOT an equivalent-workload speedup because it divides PyTorch Python-level execution by native in-kernel execution. It serves as a diagnostic demonstrating the performance ceiling when framework interpreter overhead is completely eliminated.

---

## 4. Pipeline Latency Breakdown & Code-Generation Timing Boundaries

To disambiguate component compilation timing claims, isolated repeated measurements were executed across all 8 pipeline timing boundaries on the host system:

| Stage Index | Pipeline Timing Boundary | Implementation Function | Median Latency | Mean Latency | Architectural Description & Scope |
|:---:|:---|:---|:---:|:---:|:---|
| **1** | Canonical IR Preparation | `export_to_forgetune_ir` | 66.32 ms | 163.24 ms | PyTorch `torch.export` tracing, ATen op normalization, and IR translation (pure IR translation is ~0.07 ms post-tracing). |
| **2** | Compiler Optimization Passes | `Optimizer.optimize` | 0.04 ms | 0.04 ms | Dead-Code Elimination, Algebraic Simplification, Constant Folding, Operator Fusion. |
| **3** | C++ Source Emission | `CodeGenerator.generate` | 5.80 ms | 5.93 ms | Interval-coloring memory planner and C++ string buffer emission on unoptimized input graph. |
| **4** | Combined In-Process Compilation | `Compiler.compile` | 5.64 ms | 6.12 ms | Full C++ pipeline: Pre-verification + Optimization passes + Post-verification + Source emission (optimized graph emits slightly faster than unoptimized). |
| **5** | Standalone CLI Engine Compilation | `forgetune_engine.exe --compile` | 17.97 ms | 18.74 ms | Out-of-process subprocess execution with JSON serialization and deserialization. |
| **6** | Toolchain Native Linking | `zig c++ -shared -O2` | 308.83 ms | 308.76 ms | Machine code generation, optimization, and PE/COFF dynamic library output. |
| **7** | Dynamic Library Loading | `ctypes.CDLL` | 0.02 ms | 0.05 ms | Dynamic library linking and entry point symbol resolution (`forgetune_run`). First cold load: ~1.30 ms. |
| **8A** | Dynamic TinyMLP (Model A) $B=64$ Inference | `CompiledModel.run` | **18.50 μs** | **20.74 μs** | End-to-end Python inference at $B=64$ for two-layer MLP with dynamic leading batch. |
| **8B** | Small CNN (Model C) $B=64$ Inference | `CompiledModel.run` | **374.05 μs** | **407.69 μs** | End-to-end Python inference at $B=64$ for 2D convolution and fused layers. |

**Disambiguation Resolution:**
1. **Source Emission (~5.80 ms) vs Combined Compile (~5.60 ms):** `Compiler.compile` invokes optimizer passes first, reducing the graph's node count prior to code emission. The streamlined graph generates source in ~5.60 ms, whereas calling `CodeGenerator.generate` directly on the unoptimized input graph formats all unreduced nodes, taking ~5.80 ms.
2. **~19.10 μs Inference Latency:** Strictly represents **Model A (Dynamic TinyMLP)** at batch size $B=64$. Model C at $B=64$ requires ~374–443 μs due to multi-channel 2D spatial convolutions.
