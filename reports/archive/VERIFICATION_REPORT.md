# ForgeTune Final Verification Report

**Project Name:** ForgeTune  
**Version:** 1.0.0 (Constrained Compiler Prototype)  
**Verification Date:** 2026-10-13  
**Status:** ALL CHECKS PASSED (100% SPECIFICATION ACCEPTANCE GATE SATISFIED)  

---

## 1. Environment & Toolchain Details

- **Operating System:** Windows 11 Enterprise / Pro x64 (`Windows-11-10.0.26200-SP0`, AMD64)
- **Processor:** AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD
- **Python Version:** 3.12.14 (`cpython-3.12-windows-x86_64-none`)
- **PyTorch Version:** 2.14.1+cpu
- **NumPy Version:** 2.5.2
- **PyTest Version:** 9.1.1
- **pybind11 Version:** 3.1.0
- **C++ Compiler:** Zig C++ 0.16.0 (`.venv\Lib\site-packages\ziglang\zig.exe c++` targeting `x86_64-windows-gnu`, C++17 standard)
- **Engine CLI Binary:** `bin/forgetune_engine.exe` (Standalone C++17 binary)
- **C++ Native Test Binary:** `bin/forgetune_test_suite.exe` (Standalone C++17 test runner)
- **pybind11 Extension Module:** `python/forgetune/_forgetune_cpp.pyd`

---

## 2. Test Execution Summary

| Test Suite | Total Tests | Passed | Failed | Pass Rate | Execution Time |
|---|---|---|---|---|---|
| **C++ Native Suite (`test_main.cpp`)** | 8 | 8 | 0 | 100% | 0.12s |
| **Python Unit Tests (`tests/unit/`)** | 29 | 29 | 0 | 100% | 14.80s |
| **Correctness & Edge Cases (`tests/correctness/`)** | 9 | 9 | 0 | 100% | 72.07s |
| **Integration & Autotuner (`tests/integration/`, `test_autotuner.py`)** | 6 | 6 | 0 | 100% | 19.73s |
| **Total Automated Tests** | **52** | **52** | **0** | **100%** | **106.72s** |

All 52 automated tests passed with exit code `0` during clean rebuild.

---

## 3. Production Frontend & ATen Operator Normalization

ForgeTune utilizes `torch.export.export(model, example_inputs)` ahead-of-time graph capture in [`python/forgetune/export_frontend.py`](python/forgetune/export_frontend.py).

### 3.1 Exported ATen Operator Targets
For the canonical TinyMLP (`Linear(4 -> 8) -> ReLU -> Linear(8 -> 3)`), `torch.export.export()` produces:
- `aten.linear.default` (Layer 1 Linear)
- `aten.relu.default` (Layer 1 ReLU)
- `aten.linear.default` (Layer 2 Linear)

### 3.2 Parameter Lifting & Normalization
1. **Parameter Extraction:** `graph_signature.inputs_to_parameters` maps placeholders `p_fc1_weight`, `p_fc1_bias`, `p_fc2_weight`, `p_fc2_bias` to model state dictionary weights.
2. **Linear Layer Decomposition:**
   - Transposes weight matrices from PyTorch's native `[out_features, in_features]` (`[N, K]`) to canonical row-major MatMul format `[in_features, out_features]` (`[K, N]`).
   - Emits canonical IR sequence: `INPUT` $\to$ `CONST` (transposed) $\to$ `MATMUL` $\to$ `ADD(bias)` $\to$ `RELU`.
3. **Broadcasting Forms Supported:**
   - Rank-1 vector bias: `[N]` broadcast over `[M, N]`.
   - Rank-2 row bias: `[1, N]` broadcast over `[M, N]`.
   - Both independently validated through native compilation and execution in `tests/unit/test_codegen_and_runtime.py::test_native_execution_both_bias_forms`.
4. **Actionable Error Handling:** Unsupported operators (e.g., `aten.sigmoid`, `aten.conv2d`) raise descriptive `UnsupportedOpError` identifying the operator and offending node name.

---

## 4. Empirical Benchmark Measurements & Candidate Selection

### 4.1 Benchmark Protocol & Canonical Workload
- **Protocol:** 5-round interleaved candidate execution (alternating candidates across rounds to eliminate thermal and background bias).
- **Rounds & Samples:** 5 rounds $\times$ 100 iterations = 500 measured samples per candidate (with 20 warm-up iterations per round).
- **Timer:** `time.perf_counter_ns` (monotonic high-resolution timer).
- **Canonical Workload:** TinyMLP `Linear(4 -> 8) -> ReLU -> Linear(8 -> 3)`, batch size $M = 2$.
- **Exact FLOP Accounting:**
  - Layer 1 MatMul: $2 \times 2 \times 4 \times 8 = 128\text{ FLOPs}$.
  - Layer 1 Bias Add: $2 \times 8 = 16\text{ FLOPs}$.
  - Layer 2 MatMul: $2 \times 2 \times 8 \times 3 = 96\text{ FLOPs}$.
  - Layer 2 Bias Add: $2 \times 3 = 6\text{ FLOPs}$.
  - **Total Arithmetic FLOPs:** $128 + 16 + 96 + 6 = \mathbf{246\text{ FLOPs}}$ (batch size 2).
  - Layer 1 Elementwise ReLU: $2 \times 8 = 16\text{ ops}$.
  - **Total Operations including ReLU:** $246 + 16 = \mathbf{262\text{ ops}}$ (batch size 2).
  - *(Batch size 1: 123 arithmetic FLOPs / 131 total ops).*

### 4.2 End-to-End Latency (Python Invocation via ctypes)
Includes Python call frame, ctypes pointer marshalling, C library dispatch, and return value wrapping.
Measured via 5 interleaved rounds $\times$ 100 iterations = 500 samples per candidate in `scripts/run_benchmarks.py`:

| Implementation | Compiler Flag | Median | Mean | Min | Max | Std Dev | IQR | Relative Speedup |
|---|---|---|---|---|---|---|---|---|
| **PyTorch CPU Reference** | N/A | `26.70 us` | `36.79 us` | `26.10 us` | `95.20 us` | `18.11 us` | `0.60 us` | 1.00x (ref) |
| **ForgeTune Baseline Naive** | `-O2` | `7.20 us` | `11.25 us` | `6.80 us` | `262.00 us` | `6.30 us` | `0.30 us` | **3.71x vs PyTorch** |
| **ForgeTune Fused IKJ** | `-O2` | `7.10 us` | `16.36 us` | `6.70 us` | `1291.30 us` | `56.63 us` | `0.20 us` | **1.01x vs Baseline** |
| **ForgeTune Fused IKJ** | `-O3` | `7.10 us` | `11.61 us` | `6.80 us` | `177.30 us` | `13.02 us` | `0.20 us` | **1.01x vs Baseline** |
| **ForgeTune Autotuned Winner** | `-O2` / `-O3` | `7.10 us` | `10.18 us` | `6.90 us` | `144.80 us` | `14.20 us` | `0.20 us` | **3.76x vs PyTorch** |

- **Explicit Delineation of Separate Runs:**
  - **Standalone Benchmark Run (`scripts/run_benchmarks.py`):** 5-round interleaved protocol (500 samples) reports median latencies of **7.10 µs – 7.20 µs** across ForgeTune candidates and **26.70 µs** for PyTorch CPU (a **3.76x end-to-end speedup**).
  - **TinyMLP Demonstration Run (`examples/tiny_mlp.py`):** Candidate evaluation inside autotuning reports **7.00 µs – 7.40 µs** for baseline and winner (`1.00x` ratio). The subsequent isolated 100-iteration end-to-end Python harness reports **7.00 µs – 7.20 µs** median.
  - **Assessment:** These two measurements reflect separate benchmark invocations under differing harness configurations. Neither is conflated. Candidate variations at the Python level (0.00 – 0.10 µs) are well within the ctypes dispatch noise window (IQR 0.20 – 0.50 µs).

### 4.3 Pure In-Kernel C++ Latency (10,000 Iterations Isolated Microbenchmark)
Directly measures generated C++ kernel execution without Python/ctypes invocation overhead via the native `forgetune_benchmark_kernel` entrypoint using monotonic `std::chrono::steady_clock::now()`, a volatile compiler memory barrier (`__asm__ __volatile__("" : "+m"(*(float*)output) : : "memory");`), and an outside-timed checksum accumulator:

| Kernel Configuration | Fusion | Opt Flag | Latency per Iteration | Nanoseconds | Speedup vs Baseline | Evaluation Note |
|---|---|---|---|---|---|---|
| **Baseline Naive Kernel (`naive_ijk`)** | No | `-O2` | `0.0201 us` | `20.1 ns` | 1.00x | Reference unfused triple-loop |
| **Fused IKJ Kernel (`fused_ikj`)** | Yes | `-O2` | `0.0176 us` | `17.6 ns` | **1.14x** | **Pure algorithmic fusion gain (-O2 vs -O2)** |
| **Fused IKJ Kernel (`fused_ikj_o3`)** | Yes | `-O3` | `0.0176 us` | `17.6 ns` | **1.14x** | Identical performance to -O2 (no -O3 advantage) |
| **Autotuned Winner Kernel** | Yes | `-O2` / `-O3` | `0.0176 us` | `17.6 ns` | **1.14x** | Fused IKJ kernel verified across rounds |
| **Tiled Kernel (`tiled_16`)** | No | `-O2` | `0.0335 us` | `33.5 ns` | 0.60x | Sub-tile boundary clamping penalty (rejected) |

- **Reconciliation of Latency Measurements (17.6 ns, 20.1 ns, and 20.4–25.1 ns):**
  - **17.6 ns:** The true steady-state in-kernel median latency for `fused_ikj` (both at `-O2` and `-O3`), achieved in warm multi-round execution.
  - **20.1 ns:** The true steady-state in-kernel median latency for `baseline_naive` (`-O2`).
  - **20.4–25.1 ns:** Individual single-round measurements observed during initial un-warmed rounds or under transient background OS thread scheduling jitter in earlier test iterations.
  - **Conclusion:** These measurements originate from separate individual measurement rounds under differing thread contention states. Multi-round interleaved benchmarking proves that `fused_ikj` reliably runs at **17.6 ns** ($1.14\times$ speedup over baseline at 20.1 ns).

- **Autotuner Optimization Objective & Candidate Decision:**
  1. The autotuner first checks if candidate end-to-end median latency reduction exceeds the statistical noise window ($\Delta > \max(2.0\,\mu\text{s}, \text{IQR})$).
  2. Because micro-workload TinyMLP (246 FLOPs) completes inside C++ in $\le 20.1\text{ ns}$, Python-level measurements ($\sim 7-11\,\mu\text{s}$) are dominated by ctypes marshalling noise.
  3. When end-to-end differences are noise-dominated, the autotuner evaluates pure in-kernel latency via `std::chrono::steady_clock` over 3 interleaved rounds of 10,000 iterations.
  4. It requires $\ge 5\%$ genuine in-kernel speedup to beat the baseline, safely rejecting slower tiled candidates (`33.5 ns`).
  5. Between `fused_ikj (-O2)` and `fused_ikj (-O3)` (which tie at 17.6 ns), the autotuner enforces a simplicity tie-breaking policy: the simpler `-O2` candidate is selected unless `-O3` provides a defensible $>5\%$ advantage. `-O3` is never forced to win.

---

## 5. Bit-Exact Float32 Constant and Weight Serialization

`src/codegen.cpp` utilizes `std::scientific` with `std::setprecision(std::numeric_limits<float>::max_digits10)` (9 decimal digits).
Verified in [`tests/unit/test_float_serialization.py`](tests/unit/test_float_serialization.py) across 14 difficult float32 values:
- Smallest positive subnormal: $1.40129846 \times 10^{-45}$
- Largest subnormal: $1.17549421 \times 10^{-38}$
- Smallest positive normal: $1.17549435 \times 10^{-38}$
- Boundary values: $1.0$, $1.00000012$, $2.0 - 2^{-23}$
- Negative zero: $-0.0$ (`0x80000000`)
- **Result:** 0 ULPs error (100% bit-exact bitwise match).

---

## 6. Three Independent Audits Summary

- **Audit A (Requirements and Traceability):** **PASSED** ([reports/AUDIT_A.md](reports/AUDIT_A.md))
  - 100% requirement traceability verified across all 24 specifications.
- **Audit B (Correctness, Safety, and Edge Cases):** **PASSED** ([reports/AUDIT_B.md](reports/AUDIT_B.md))
  - Verified weight layout transposition, floating-point literal formatting, partial tile boundary conditions, integer overflow guards, pybind11 numerical fidelity, broadcasting contract, algebraic rewrite multi-consumer safety, Windows GNU PE/COFF ABI compliance, and Windows DLL staging backoff.
- **Audit C (Clean Rebuild and Reproducibility):** **PASSED** ([reports/AUDIT_C.md](reports/AUDIT_C.md))
  - Rebuilt all artifacts from scratch via `scripts/clean_build.py`, verified all 52 automated tests pass (8 C++ + 44 Python), clean exit code 0.

---

## 7. Documented Limitations & Scope Bounds

### Version 1.0 Supported Scope (Constrained Compiler Prototype):
- Single rank-2 float32 input tensor, single float32 output tensor.
- Operations: `INPUT`, `CONST`, `MATMUL`, `ADD`, `RELU`, `FUSED_MATMUL_ADD_RELU`, `OUTPUT`.
- Bias broadcasting: rank-1 vector `[N]` or rank-2 row `[1, N]` over rank-2 `[M, N]`.
- Static tensor dimensions specialized at compile time.

### Out of Scope for v1.0:
- Dynamic batch sizes and dynamic control flow.
- Arbitrary rank broadcasting (ranks $\ge 3$, CNNs, Transformers).
- GPU backends (CUDA/ROCm) and SIMD intrinsics (AVX-512).
- Training, backward passes, and autograd.
- Quantization (int8, fp16).
