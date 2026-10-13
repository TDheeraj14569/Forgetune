# ForgeTune

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/Standard-C%2B%2B17-blue.svg)](CMakeLists.txt)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)

ForgeTune is a specification-driven, correctness-checked, self-optimizing machine-learning compiler built from scratch. It extracts computation graphs from PyTorch models via `torch.export`, normalizes them into custom C++ Intermediate Representation (IR), verifies graph structure, applies graph optimization passes (constant folding, algebraic simplification, dead-code elimination, and operator fusion), generates standalone C++ CPU kernels across multiple variants (naive, cache-friendly loop ordering, 2D tiling, and fused loop nests), compiles them into shared libraries, and autotunes kernel selection through an empirical correctness gate on the host machine.

ForgeTune implements its own independent compiler engine and does not delegate tensor arithmetic to TVM, MLIR, LLVM APIs, TorchInductor, TorchScript, Triton, or any third-party compiler.

---

## 1. Key Features

- **Frontend:** Ahead-of-time graph extraction via `torch.export.export`. Normalizes ATen operations, correctly transposing dense layer weights to canonical row-major layout `[K, N]`.
- **Custom C++17 IR & Graph Verifier:** Independent source-of-truth IR with static types, tensor shapes, stable IDs, and use counts. Comprehensive verifier rejects malformed graphs, dimension mismatches, and cycles with clear diagnostics.
- **Independent Reference Interpreter:** Evaluates IR graphs using straightforward, auditable C++ and Python arithmetic as an unambiguous correctness baseline.
- **Modular Pass Manager:**
  - **Constant Folding:** Compile-time evaluation of constant subgraphs.
  - **Algebraic Simplification:** Safe algebraic rewrites (e.g., additive zero identity, ReLU idempotency).
  - **Dead-Code Elimination:** Backward reachability pruning of unused operations.
  - **Operator Fusion:** Pattern matches `MATMUL -> ADD -> RELU` into a single fused kernel, eliminating intermediate memory buffers.
- **Standalone C++ Code Generator:** Emits clean, readable C++ code with zero external tensor dependencies. Features naive `i, j, k`, cache-friendly `i, k, j`, 2D tiled matrix multiplication with boundary handling, and fused kernel generators.
- **Correctness-Gated Autotuner:** Evaluates candidates across multiple input distributions (zeros, negatives, random scales), rejecting any candidate failing strict numerical tolerance (`atol=1e-4`, `rtol=1e-4`), and selecting the fastest implementation.
- **High-Resolution Benchmark Harness:** Uses monotonic timers (`perf_counter_ns`), reports median latency and IQR dispersion, and records machine metadata.

---

## 2. Installation & Prerequisites

### Prerequisites
- Python 3.11 or newer (tested with Python 3.12.14)
- PyTorch >= 2.1 (tested with PyTorch 2.14.1 CPU)
- NumPy >= 1.24
- C++17 Compiler (Zig C++, Clang, GCC, or MSVC)

### Setup
```bash
# Create virtual environment
uv venv .venv --python 3.12
.venv\Scripts\activate

# Install dependencies
uv pip install torch numpy pytest ziglang
```

---

## 3. Quickstart Example (TinyMLP)

Run the end-to-end TinyMLP example:
```bash
python examples/tiny_mlp.py
```

### Python API Usage
```python
import torch
import torch.nn as nn
import forgetune

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = TinyMLP().eval()
x = torch.randn(2, 4)

# Compile model with autotuning
compiled = forgetune.compile_model(
    model=model,
    example_inputs=(x,),
    output_dir="build/artifacts/tiny_mlp",
    autotune=True
)

# Run native CPU inference
output = compiled.run(x)
print("Output:", output)

# View optimization and autotuning report
print("Report:", compiled.optimization_report())
```

---

## 4. Full Verification Suite

Run all verification checks (C++ native tests, Python tests, E2E integration):
```bash
python scripts/verify_all.py
```

Run benchmarks comparing PyTorch CPU vs ForgeTune:
```bash
python scripts/run_benchmarks.py
```

Clean build:
```bash
python scripts/clean_build.py
```

---

## 5. Repository Layout

```text
ForgeTune/
├── CMakeLists.txt
├── CMakePresets.json
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── include/
│   └── forgetune/
│       ├── ir.hpp
│       ├── graph_verifier.hpp
│       ├── interpreter.hpp
│       ├── pass_manager.hpp
│       ├── optimizer.hpp
│       ├── fusion.hpp
│       ├── kernel_config.hpp
│       ├── codegen.hpp
│       ├── json_io.hpp
│       └── compiler.hpp
│
├── src/
│   ├── ir.cpp
│   ├── graph_verifier.cpp
│   ├── interpreter.cpp
│   ├── pass_manager.cpp
│   ├── optimizer.cpp
│   ├── fusion.cpp
│   ├── codegen.cpp
│   ├── json_io.cpp
│   └── main.cpp
│
├── python/
│   └── forgetune/
│       ├── __init__.py
│       ├── export_frontend.py
│       ├── graph_adapter.py
│       ├── compiler.py
│       ├── native_compiler.py
│       ├── autotuner.py
│       ├── benchmark.py
│       ├── runtime.py
│       ├── manifest.py
│       └── errors.py
│
├── examples/
│   ├── tiny_mlp.py
│   ├── compile_and_run.py
│   └── inspect_export.py
│
├── tests/
│   ├── cpp/test_main.cpp
│   ├── unit/
│   │   ├── test_operators.py
│   │   ├── test_ir.py
│   │   ├── test_optimizations.py
│   │   └── test_codegen_and_runtime.py
│   ├── correctness/
│   │   ├── test_differential.py
│   │   └── test_edge_cases.py
│   ├── integration/
│   │   └── test_e2e_tinymlp.py
│   └── test_autotuner.py
│
├── scripts/
│   ├── verify_all.py
│   ├── clean_build.py
│   └── run_benchmarks.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── IR_SPECIFICATION.md
│   ├── OPTIMIZATIONS.md
│   ├── BENCHMARK_METHODOLOGY.md
│   └── TROUBLESHOOTING.md
│
└── reports/
    ├── benchmark_report.json
    ├── AUDIT_A.md
    ├── AUDIT_B.md
    ├── AUDIT_C.md
    └── VERIFICATION_REPORT.md
```
