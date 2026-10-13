# ForgeTune v2.0 Benchmarking Guide & Methodology

This document outlines the benchmarking methodology, timing protocols, and performance metrics utilized in ForgeTune v2.0.

---

## 1. Disambiguation of Latency Metrics

To prevent synthetic speedup inflation or comparing disparate execution paths, ForgeTune strictly separates three latency boundaries:

1. **Pure In-Kernel Execution (`forgetune_benchmark_kernel`):**
   - **Mechanism:** Implemented entirely in C++ inside the compiled shared library.
   - **Clock:** C++17 `std::chrono::steady_clock` with memory barriers (`__asm__ __volatile__("" : "+m"(*(float*)output) : : "memory");`).
   - **Purpose:** Measures raw hardware compute latency of the generated C++ loop nest, cache residency, and SIMD execution. Bypasses Python interpreter overhead, argument marshalling, and ctypes call frames.
   - **Autotuning Objective:** Autotuner candidates are ranked by this metric because it isolates genuine compiler code-generation differences from host Python noise.

2. **End-to-End Native Inference (`CompiledModel.benchmark`):**
   - **Mechanism:** Invokes `CompiledModel.run()` from Python using representative numpy arrays.
   - **Clock:** Python `time.perf_counter_ns()` with monotonic nanosecond resolution.
   - **Purpose:** Measures real-world inference latency from a Python caller, including numpy contiguous array verification, ctypes argument conversion, DLL function dispatch, and output buffer allocation.

3. **PyTorch Eager CPU Baseline (`measure_pytorch_latency`):**
   - **Mechanism:** Invokes the original `torch.nn.Module` forward pass in Python under `torch.no_grad()`.
   - **Purpose:** Represents the standard eager-mode CPU execution baseline for comparison.

---

## 2. Statistical Protocol & Anti-Bias Measures

1. **Warm-Up Discard:** 25 warm-up iterations are executed before measurement to bring the CPU instruction and data caches to steady state and mitigate initial frequency scaling artifacts.
2. **Measurement Sample Size:** 100 to 500 trials executed sequentially.
3. **Statistical Dispersion:**
   - **Median:** Primary reporting metric, robust to operating-system context switches and thread interrupts.
   - **IQR:** Interquartile range ($Q_{75} - Q_{25}$) to characterize variance.
   - **Dispersion Reporting:** Min, Max, Mean, Standard Deviation, P90, and P99 percentiles are recorded in `v2_benchmark_report.json`.
4. **Interleaved Candidate Evaluation:** In the autotuner, candidates are evaluated in interleaved rounds rather than consecutively, eliminating initial-candidate cache or thermal throttling bias.

---

## 3. Running Benchmarks

Execute the standalone multi-model benchmark suite:
```powershell
$env:PYTHONPATH='python;.'
python scripts/run_v2_benchmarks.py
```
Outputs:
- `reports/v2_benchmark_report.json`: Machine-readable timing samples and metadata.
- `reports/V2_BENCHMARK_REPORT.md`: Formatted markdown tables with measured speedups across batch sizes.
