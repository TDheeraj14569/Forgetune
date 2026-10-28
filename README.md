# ForgeTune v2.0

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Standard](https://img.shields.io/badge/Standard-C%2B%2B17-blue.svg)](CMakeLists.txt)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/Tests-229%2F229%20PASS-brightgreen.svg)](reports/V2_REQUIREMENTS_MATRIX.md)
[![Verification](https://img.shields.io/badge/Protocol-3--Round%20Bitwise%20Exact-brightgreen.svg)](reports/V2_FINAL_VERIFICATION.md)

**ForgeTune v2.0** is an ahead-of-time (AOT) C++17 machine learning inference compiler and lightweight native runtime designed for low-latency CPU inference with dynamic batching and zero steady-state heap allocations.

---

## 1. The Engineering Problem

In edge, embedded, and latency-critical CPU inference, executing small-to-medium neural networks through standard deep learning frameworks incurs substantial overhead:
- **Interpreter & Dispatch Overhead:** Dynamic framework runtimes introduce traversal latency, tensor metadata allocation, and operator dispatch costs that often exceed the computation time of the inner linear algebra operations.
- **Fragmented Memory Lifetimes:** Naive per-layer allocation results in recurring heap allocations, cache pollution, and memory fragmentation.
- **Heavyweight Runtimes:** Full runtime dependencies (such as large LLVM-based JIT compilers or extensive C++ framework runtimes) add binary bloat and deployment complexity in constrained environments.

ForgeTune solves this by extracting PyTorch computational graphs into an independent, strongly typed C++17 intermediate representation (IR), applying domain-specific compiler optimizations (graph rewriting, multi-operator loop fusion, and interval-coloring memory planning), and compiling self-contained native shared libraries (`.dll`) that execute with zero framework dependencies.

---

## 2. Architecture & Compilation Pipeline

ForgeTune implements an independent compiler and native runtime without delegating tensor arithmetic to TVM, MLIR, LLVM APIs, TorchInductor, TorchScript, or Triton.

```mermaid
flowchart TD
    subgraph Frontend["1. Frontend & Ingestion"]
        A["PyTorch nn.Module"] -->|"torch.export.export()<br/>(Dynamic Batch B in [1, 128])"| B["FX Graph Module"]
        B -->|"export_to_forgetune_ir()"| C["ForgeTune IR Graph<br/>(Schema 2.0)"]
    end

    subgraph Optimization["2. C++ Optimization Passes"]
        C --> D["Graph Verifier<br/>(DAG Acyclicity & Type Inference)"]
        D --> E["Optimization Pipeline"]
        E --> E1["Constant Folding"]
        E --> E2["Algebraic Simplification"]
        E --> E3["Dead Code Elimination (DCE)"]
        E --> E4["Operator Fusion<br/>(Conv2D+ReLU, MatMul+Add+ReLU)"]
    end

    subgraph Memory["3. Memory Planning"]
        E4 --> F["Liveness Analysis & Memory Planner<br/>(Topological Interval Coloring)"]
        F -->|"42.9% - 60.0% Peak Buffer Reduction"| G["Static Arena Offsets<br/>(64-byte AVX-512 Aligned)"]
    end

    subgraph Native["4. Native Codegen & Runtime"]
        G --> H["C++17 Standalone Codegen<br/>(Dynamic Batch Indexing)"]
        H --> I["Zig C++ / Native Toolchain"]
        I --> J["Native Shared Library (.dll)<br/>(Dual Versioned C ABI)"]
        J --> K["NativeRuntime (ctypes / pybind11)<br/>(Zero Steady-State Heap Allocation)"]
    end
```

---

## 3. Core Technical Capabilities

### Strongly Typed Intermediate Representation (IR Schema 2.0)
- Explicit node definitions (`IRNode`, `IRGraph`) tracking topological producers, consumers, and data types (`FLOAT32`).
- Symbolic dynamic dimensions supporting leading batch variation ($B \in [1, B_{max}]$) alongside static rank-2 ($[B, K]$) and rank-4 ($[B, C, H, W]$) tensor geometries.
- Verification pass enforcing strict DAG acyclicity, channel compatibility, and non-negative spatial dimensions.

### Optimization & Operator Fusion
- **Multi-Operator Loop Fusion:** Fuses `CONV2D + RELU` and `MATMUL + ADD + RELU` into unified cache-resident loops, eliminating intermediate activation store/reload cycles.
- **Algebraic Rewriting:** Simplifies identity arithmetic ($x + 0 \to x$) and idempotent activations ($\text{ReLU}(\text{ReLU}(x)) \to \text{ReLU}(x)$).
- **Constant Folding & DCE:** Pre-computes weight transformations and removes unreferenced dead subgraphs at compile time.

### Liveness-Based Intermediate Buffer Reuse
- Analyzes activation birth and death intervals across topological execution steps.
- Solves buffer allocation using interval-coloring register-allocation principles.
- Packs all disjoint activation lifetimes into a single pre-allocated `AlignedArenaPool` (64-byte aligned for AVX2/AVX-512 vectorization), reducing intermediate activation storage by **42.9% to 60.0%**.

### Dynamic Batch Execution ($B \in [1, 128]$)
- Emits C++ inner loops parameterized on batch length $B$ without requiring recompilation per batch size.
- Hardened C ABI boundary validation: out-of-bounds requests ($B \le 0$ or $B > B_{max}$) return `-2` and trigger Python `InvalidShapeError` before memory access (with `-1` for null pointers, `-3` for output mismatch, and `-4` for internal runtime exceptions).

### Dual Versioned C ABI
- Defined in `include/forgetune/forgetune_abi.h`:
  - Dynamic Batch V2 Entry Point: `int forgetune_run_v2(const float* inp, size_t inp_len, float* out, size_t out_len, int64_t batch_size)`
  - Legacy Static V1 Entry Point: `int forgetune_run_v1(const float* inp, size_t inp_len, float* out, size_t out_len)`
  - Query Exports: `int forgetune_is_dynamic(void)`, `int64_t forgetune_max_batch(void)`

---

## 4. Empirical Benchmarks & Performance Analysis

All benchmarks are measured on AMD64 hardware (Windows 11, PyTorch 2.14.1+cpu). Latency measurements are reported using authoritative on-disk artifacts (`reports/v2_benchmark_report.json` and `reports/v2_paired_b64_benchmark_data.json`).

### Multi-Model Benchmark Table (Dataset: `v2-bench-1792863787-0233951b`)

| Workload Model | Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8T)** | **Controlled E2E Speedup (1T)** | ForgeTune Native Kernel (μs) | **Native Kernel Speedup** | E2E - Kernel Residual (μs) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model A (Dynamic TinyMLP)** | $B=1$ | 41.90 | 39.40 | 8.70 | **4.82x** | **4.53x** | 0.289 | **1.00x** | 8.41 |
| *Linear(16->32) -> ReLU -> Linear(32->8)* | $B=4$ | 45.85 | 37.35 | 13.50 | **3.40x** | **2.77x** | 1.143 | **1.00x** | 12.36 |
| *(Selected: `baseline_naive`)* | $B=16$ | 55.70 | 27.70 | 16.40 | **3.40x** | **1.69x** | 3.367 | **1.00x** | 13.03 |
| | $B=64$ | 49.90 | 44.20 | 22.10 | **2.26x** | **2.00x** | 16.937 | **1.00x** | 5.16 |
| **Model B (Residual MLP)** | $B=1$ | 41.80 | 41.80 | 13.60 | **3.07x** | **3.07x** | 0.117 | **1.00x** | 13.48 |
| *Linear + Residual Skip + ReLU* | $B=4$ | 43.60 | 44.30 | 9.20 | **4.74x** | **4.82x** | 0.394 | **1.00x** | 8.81 |
| *(Selected: `baseline_naive`)* | $B=16$ | 45.60 | 45.20 | 10.30 | **4.43x** | **4.39x** | 1.446 | **1.00x** | 8.85 |
| | $B=64$ | 77.10 | 50.50 | 14.40 | **5.35x** | **3.51x** | 6.057 | **1.00x** | 8.34 |
| **Model C (Small CNN)** | $B=1$ | 84.05 | 88.45 | 16.80 | **5.00x** | **5.26x** | 7.986 | **1.18x** | 8.81 |
| *Conv2d -> ReLU -> Conv2d -> ReLU -> FC* | $B=4$ | 321.70 | 307.60 | 37.60 | **8.56x** | **8.18x** | 26.789 | **1.26x** | 10.81 |
| *(Selected: `loop_ikj`)* | $B=16$ | 352.00 | 286.90 | 89.65 | **3.93x** | **3.20x** | 92.864 | **1.30x** | -3.21\* |
| | $B=64$ | 409.53 | 462.02 | 442.70 | **0.93x** | **1.04x** | 421.300 | **1.10x** | 24.13 |

*\*Residual Note: For Model C at $B=16$, the negative residual (-3.21 μs) is a known sequential timing artifact resulting from CPU frequency scaling across non-interleaved benchmark loops. To resolve this, interleaved paired testing was conducted at $B=64$.*

### Statistically Rigorous Paired Hypothesis Testing ($N=100$ Blocks)

To evaluate Model C at $B=64$ without sequential bias, an interleaved 100-block paired experiment was executed with alternating execution order (`reports/v2_paired_b64_benchmark_data.json`):
- **PyTorch 1-Thread E2E Median:** $361.98\ \mu\text{s}$
- **ForgeTune E2E Median:** $343.03\ \mu\text{s}$
- **Mean Paired Difference ($\bar{d} = t_{PT1} - t_{FT}$):** $+23.773\ \mu\text{s}$ (ForgeTune is faster)
- **Standard Error ($SE$):** $4.289\ \mu\text{s}$ ($df = 99$)
- **One-Sided Non-Inferiority ($H_0: \mu_d \le -18.10\ \mu\text{s}$):** $t = 9.762$, exact Student's t $p = 1.81 \times 10^{-16} \ll 0.0001$. Non-inferiority is certified.
- **Two-Sided TOST Equivalence ($\pm 5\%$ margin):** Inconclusive strictly because ForgeTune outperforms PyTorch 1T ($90\%\ \text{CI} = [+16.65, +30.89]\ \mu\text{s}$), placing the upper confidence bound above $+18.10\ \mu\text{s}$.
- **Bootstrap 90% Confidence Intervals ($B=10,000$ resamples):**
  - Mean Difference: $[+17.03\ \mu\text{s},\; +30.91\ \mu\text{s}]$
  - Median Difference: $[+14.15\ \mu\text{s},\; +21.70\ \mu\text{s}]$

---

## 5. Quickstart & Usage

### Prerequisites
- Python 3.11 or 3.12
- C++17 compatible toolchain (Zig C++ bundled via `ziglang` Python package, or Clang/GCC/MSVC)

### Installation
```powershell
# Clone repository
git clone https://github.com/TDheeraj14569/Forgetune.git
cd Forgetune

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install required dependencies
pip install torch numpy pytest ziglang pybind11
```

### Compiling and Running a Dynamic Model
```python
import torch
import torch.nn as nn
import forgetune

# 1. Define PyTorch model
class DynamicTinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = DynamicTinyMLP().eval()

# 2. Compile model to standalone C++ DLL with dynamic batching
compiled_model = forgetune.compile_model(
    model=model,
    example_inputs=(torch.randn(2, 16),),
    output_dir="build/artifacts/tinymlp",
    autotune=True,
    dynamic_batch=True,
    enable_fusion=True
)

# 3. Single compiled artifact executes any batch size in [1, 128]
out_b1  = compiled_model.run(torch.randn(1, 16))   # Batch 1
out_b16 = compiled_model.run(torch.randn(16, 16))  # Batch 16
out_b64 = compiled_model.run(torch.randn(64, 16))  # Batch 64

print(f"Batch 1 output shape:  {out_b1.shape}")
print(f"Batch 64 output shape: {out_b64.shape}")
```

---

## 6. Verification & Test Suite Accounting

The repository enforces a defensible, multi-tier automated test suite:

```powershell
# 1. Run full Python test suite (216 tests)
.\.venv\Scripts\python.exe -m pytest -v tests/

# 2. Run standalone native C++ unit tests (13 tests)
.\bin\forgetune_test_suite.exe

# 3. Execute master 3-round verification protocol
.\.venv\Scripts\python.exe scripts/v2_verify_all.py --mode=full
```

### Test Accounting Summary
- **Total Enumerated Tests:** **229 tests** (100% passing, 0 failed, 0 skipped, 0 blocked).
  - **216 Python Tests:** Collected and executed via `pytest`. Covers operator semantics, algebraic simplification, numerical stress, dual-ABI contracts, regression suites for identity models, tile boundaries, commutative broadcast, multi-process determinism, and 25 audit defect regressions.
  - **13 Native C++ Tests:** Standalone unit tests executing without Python runtime dependencies in `bin/forgetune_test_suite.exe`.
  - *Note on overlap:* All 13 native C++ test functions are also executed in-process within pytest via `tests/unit/test_pybind11_bindings.py::test_pybind11_native_cpp_test_suite`.
- **Compiler Diagnostics:** Builds with **0 warnings and 0 errors** under `-target x86_64-windows-gnu -I include -std=c++17 -O2 -Wall -Wextra`.
- **Bitwise Reproducibility:** 3-round verification protocol compares 65,536 float32 values across independent clean compilations with **0 bit divergences** (0/2,097,152 bits).

---

## 7. Scope & Current Limitations

To maintain architectural transparency, the current scope and limitations are explicitly documented:
- **Execution Target:** CPU inference only. No CUDA, Metal, or GPU backends are implemented.
- **Operator Set:** Designed for MLPs and Feed-Forward CNNs. Supports `MATMUL`, `ADD`, `MUL`, `RELU`, `SIGMOID`, `TANH`, `CONV2D`, `FLATTEN`, and fused combinations. Does not support dynamic control flow, RNNs, or Multi-Head Attention mechanisms.
- **Dynamic Dimension Scope:** Dynamic behavior is strictly constrained to the leading batch dimension ($B \le 128$). Internal tensor channel and spatial dimensions ($C, H, W$) are resolved statically at export time.
- **Threading Model:** Single-threaded execution per runtime instance. Multi-threading is achieved at the application level via concurrent independent runtime instances.

---

## 8. Technology Stack

- **Core Compiler & Engine:** Modern C++17, STL, custom AST & IR DAG.
- **Compilation Toolchains:** Zig C++ (LLVM backend), Clang, GCC, MSVC.
- **Frontend & Graph Capture:** PyTorch 2.14+, `torch.export` (AOT FX graph tracing).
- **Bindings & ABI:** C ABI (`extern "C"`), `ctypes`, `pybind11`.
- **Testing & Verification:** `pytest`, `numpy`, `scipy.stats` (hypothesis testing & bootstrap estimation).
- **CI/CD:** GitHub Actions (`windows-latest` automated matrix).

---

## 9. Technical Reports & Audit Documentation

For detailed architectural, mathematical, and forensic logs:
- [Requirements Traceability Matrix (REQ-01 to REQ-27)](reports/V2_REQUIREMENTS_MATRIX.md)
- [Benchmark Evaluation & Statistical Analysis](reports/V2_BENCHMARK_REPORT.md)
- [Intermediate-Buffer Storage Reduction Report](reports/V2_MEMORY_REPORT.md)
- [Master Three-Round Verification Report](reports/V2_FINAL_VERIFICATION.md)
- [Authoritative 12-Gate Closure Audit](reports/V2_FINAL_CLOSURE_AUDIT.md)
- [Windows Application Control Forensic Diagnostic](reports/V2_SAC_DIAGNOSTIC.md)
