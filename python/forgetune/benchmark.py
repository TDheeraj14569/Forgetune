"""
High-resolution benchmarking harness for ForgeTune.
Measures warm-up and steady-state inference latency using monotonic clocks,
calculates statistical dispersion, and records machine metadata.
"""

import time
import platform
from typing import Dict, Any, Callable
import numpy as np

def benchmark_function(
    fn: Callable[[], Any],
    warmup_iters: int = 10,
    measure_iters: int = 50
) -> Dict[str, Any]:
    """
    Benchmarks a callable using time.perf_counter_ns().
    Returns comprehensive latency metrics in microseconds (us).
    """
    # 1. Warm-up iterations to steady state
    for _ in range(warmup_iters):
        fn()

    # 2. Measurement iterations
    timings_ns = []
    for _ in range(measure_iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        timings_ns.append(t1 - t0)

    # Convert to microseconds
    timings_us = np.array(timings_ns, dtype=np.float64) / 1000.0

    median_us = float(np.median(timings_us))
    mean_us = float(np.mean(timings_us))
    min_us = float(np.min(timings_us))
    max_us = float(np.max(timings_us))
    std_us = float(np.std(timings_us))

    q25 = float(np.percentile(timings_us, 25))
    q75 = float(np.percentile(timings_us, 75))
    iqr_us = q75 - q25

    return {
        "warmup_iterations": warmup_iters,
        "measured_iterations": measure_iters,
        "median_latency_us": median_us,
        "mean_latency_us": mean_us,
        "min_latency_us": min_us,
        "max_latency_us": max_us,
        "std_dev_us": std_us,
        "iqr_us": iqr_us,
        "p90_latency_us": float(np.percentile(timings_us, 90)),
        "p99_latency_us": float(np.percentile(timings_us, 99)),
        "raw_samples_us": timings_us.tolist()
    }

def get_machine_metadata() -> Dict[str, Any]:
    """
    Collects host environment identification metadata.
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
