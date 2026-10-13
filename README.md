# ForgeTune v2.0

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/Standard-C%2B%2B17-blue.svg)](CMakeLists.txt)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Version](https://img.shields.io/badge/Release-v2.0.0--dev-green.svg)](reports/V2_FINAL_VERIFICATION.md)
[![Tests](https://img.shields.io/badge/Tests-162%2F162%20PASS-brightgreen.svg)](reports/V2_REQUIREMENTS_MATRIX.md)

ForgeTune v2.0 is a specification-driven, correctness-verified, self-optimizing machine-learning inference compiler built from scratch for CPU execution. It extracts computation graphs from PyTorch models via `torch.export`, normalizes them into custom C++ Intermediate Representation (IR Schema 2.0), applies graph optimizations (constant folding, algebraic simplification, dead-code elimination, and multi-operator fusion), performs liveness-based intermediate buffer reuse, generates standalone C++17 CPU kernels supporting dynamic batch execution ($B \in [1, 128]$), compiles them into shared libraries, and autotunes kernel selection through an empirical correctness gate on the host machine.

ForgeTune implements its own independent compiler engine and runtime; it does not delegate tensor arithmetic to TVM, MLIR, LLVM APIs, TorchInductor, TorchScript, Triton, or any third-party compiler.

---

## 1. Key Features in v2.0

- **Dynamic Batch Inference ($B \in [1, 128]$):** Single compiled shared library executes multiple batch sizes ($B=1, 2, 7, 16, 32, 64, 128$) with $O(1)$ dispatch overhead and runtime out-of-bounds contract validation ($B \le 0$ or $B > 128$ rejected cleanly).
- **Expanded Operator Set:**
  - `MUL`: Elementwise multiplication with scalar, vector, and 4D channel broadcasting.
  - `SIGMOID`: Numerically stable activation with saturation clamping.
  - `TANH`: Numerically stable hyperbolic tangent activation.
  - `CONV2D`: Float32 2D direct convolution supporting NCHW activations, OIHW weights, configurable stride, padding, dilation, and optional bias.
  - `FLATTEN`: Spatial dimension flattening preserving the leading batch dimension.
  - `FUSED_CONV2D_RELU`: Fuses 2D convolution with non-linear activation in a single cache-resident loop nest.
- **Liveness-Based Intermediate Buffer Reuse:**
  - Analyzes birth and death intervals of all intermediate activation buffers.
  - Packs non-overlapping activations into a single 64-byte aligned `thread_local` memory arena.
  - Reduces peak intermediate activation memory by **42.9% to 60.0%**.
- **Unified Dual C ABI:**
  - Primary dynamic inference entry point: `int forgetune_run(const float* inp, size_t inp_len, float* out, size_t out_len, int64_t batch_size)`.
  - Backward-compatible v1 wrapper: `int forgetune_run_v1(inp, inp_len, out, out_len)`.
  - Capability query exports: `int forgetune_is_dynamic(void)`, `int64_t forgetune_max_batch(void)`.
- **Three Supported Real Workload Models:**
  - **Model A (Dynamic TinyMLP):** `Linear -> ReLU -> Linear` with dynamic batching.
  - **Model B (Residual MLP):** Multi-layer MLP with residual skip connections and elementwise addition.
  - **Model C (Small CNN):** `Conv2d -> ReLU -> Conv2d -> ReLU -> Flatten -> Linear`.
- **100% Backward Compatibility:** All 119 legacy v1.0 automated unit, acceptance, differential, and robustness tests continue to pass with zero regressions.

---

## 2. Performance & Benchmark Summary

Measured on host AMD64 CPU (AMD64 Family 25 Model 117 Stepping 2, Windows 11) using `scripts/run_v2_benchmarks.py`:

| Workload Model | Batch Size ($B$) | PyTorch Eager (μs) | ForgeTune E2E (μs) | ForgeTune Native Kernel (μs) | E2E Speedup | In-Kernel Speedup |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model A (Dynamic TinyMLP)** | $B=1$ | 36.10 μs | 12.50 μs | 0.216 μs | **2.89x** | **167.3x** |
| | $B=4$ | 44.15 μs | 12.80 μs | 0.737 μs | **3.45x** | **59.9x** |
| | $B=16$ | 45.90 μs | 15.00 μs | 2.760 μs | **3.06x** | **16.6x** |
| | $B=64$ | 37.15 μs | 23.50 μs | 10.220 μs | **1.58x** | **3.6x** |
| **Model B (Residual MLP)** | $B=1$ | 42.40 μs | 8.90 μs | 0.137 μs | **4.76x** | **308.6x** |
| | $B=4$ | 45.20 μs | 14.60 μs | 0.442 μs | **3.10x** | **102.4x** |
| | $B=16$ | 46.30 μs | 9.95 μs | 3.267 μs | **4.65x** | **14.2x** |
| | $B=64$ | 113.30 μs | 21.95 μs | 8.612 μs | **5.16x** | **13.2x** |
| **Model C (Small CNN)** | $B=1$ | 177.85 μs | 31.55 μs | 17.189 μs | **5.64x** | **10.4x** |
| | $B=4$ | 442.05 μs | 82.20 μs | 69.325 μs | **5.38x** | **6.4x** |
| | $B=16$ | 335.90 μs | 273.10 μs | 251.369 μs | **1.23x** | **1.3x** |

---

## 3. Installation & Setup

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
uv pip install torch numpy pytest ziglang
```

---

## 4. Quickstart: Compiling Models in v2.0

### Model A: Dynamic TinyMLP
```python
import torch
import torch.nn as nn
import forgetune

class DynamicTinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = DynamicTinyMLP().eval()
x_example = torch.randn(2, 16)

# Compile with dynamic batching B in [1, 128]
compiled = forgetune.compile_model(
    model=model,
    example_inputs=(x_example,),
    output_dir="build/artifacts/tinymlp",
    autotune=True,
    dynamic_batch=True,
    enable_fusion=True
)

# Single compiled model executes any valid batch size:
out_1 = compiled.run(torch.randn(1, 16))   # Batch 1
out_16 = compiled.run(torch.randn(16, 16)) # Batch 16
out_64 = compiled.run(torch.randn(64, 16)) # Batch 64
```

### Model C: Small CNN
```python
import torch
import torch.nn as nn
import forgetune

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        return self.fc(self.flatten(self.relu2(self.conv2(self.relu1(self.conv1(x))))))

model = SmallCNN().eval()
compiled = forgetune.compile_model(
    model=model,
    example_inputs=(torch.randn(2, 1, 8, 8),),
    output_dir="build/artifacts/cnn",
    autotune=True,
    dynamic_batch=True,
    enable_fusion=True
)
out = compiled.run(torch.randn(4, 1, 8, 8))
```

---

## 5. Verification & Audit Suite

ForgeTune v2.0 includes automated verification and audit runners:

```bash
# 1. Run Master Three-Round Verification (Round 1, Round 2, Round 3)
python scripts/v2_verify_all.py

# 2. Run Comprehensive Multi-Model Benchmarks
python scripts/run_v2_benchmarks.py

# 3. Generate Intermediate Buffer Reuse Memory Report
python scripts/generate_v2_memory_report.py

# 4. Run Legacy Full Verification Suite
python scripts/verify_all.py
```

### Generated Audit Artifacts:
- `reports/V2_REQUIREMENTS_MATRIX.md`: 27/27 requirements verified PASS.
- `reports/V2_AUDIT_1_REQUIREMENTS.md`: Independent requirements traceability audit.
- `reports/V2_AUDIT_2_CODE_QUALITY.md`: Architecture, security, and memory safety audit.
- `reports/V2_AUDIT_3_RELEASE_READINESS.md`: Performance acceptance and release readiness audit.
- `reports/V2_FINAL_VERIFICATION.md`: Three-round verification execution logs.
- `reports/V2_BENCHMARK_REPORT.md`: Multi-model latency and speedup analysis.
- `reports/V2_MEMORY_REPORT.md`: Intermediate buffer reuse and allocation map.
