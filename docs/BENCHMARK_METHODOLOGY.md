# ForgeTune Benchmark Methodology

## 1. Timing Protocol

ForgeTune uses a rigorous, reproducible benchmarking harness designed to measure CPU inference latency without synthetic inflation:

- **Clock:** Python's monotonic clock via `time.perf_counter_ns()` with nanosecond resolution.
- **Warm-Up Iterations:** Configurable (default 20–50 runs) to eliminate JIT, CPU frequency scaling, and cache cold-start noise.
- **Measured Iterations:** Repeated trials (default 100–200 iterations) executed sequentially.
- **Statistical Dispersion:**
  - Median Latency (primary reporting metric, robust to OS scheduling spikes).
  - Minimum and Maximum Latencies.
  - Standard Deviation (`std_dev_us`).
  - Interquartile Range (`iqr_us` = Q75 - Q25).
  - 90th and 99th percentile latencies.

## 2. Separation of Concerns

- **Compilation Overhead:** Timed separately during `compile_shared_library`.
- **Autotuning Overhead:** Timed and recorded separately in `autotuning_report.json`.
- **Steady-State Inference:** Purely measures execution of the loaded native kernel `forgetune_run`.
- **No PyTorch in Native Benchmarks:** Native benchmarks invoke `ctypes.CDLL.forgetune_run` directly on raw memory buffers without passing through PyTorch or Python dispatchers.
