# ForgeTune v2.0

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Standard](https://img.shields.io/badge/Standard-C%2B%2B17-blue.svg)](CMakeLists.txt)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Windows CI](https://github.com/TDheeraj14569/Forgetune/actions/workflows/windows_verification.yml/badge.svg)](https://github.com/TDheeraj14569/Forgetune/actions/workflows/windows_verification.yml)
[![Linux CI](https://github.com/TDheeraj14569/Forgetune/actions/workflows/linux_verification.yml/badge.svg)](https://github.com/TDheeraj14569/Forgetune/actions/workflows/linux_verification.yml)
[![Tests](https://img.shields.io/badge/Tests-306%2F306%20PASS-brightgreen.svg)](reports/V2_REQUIREMENTS_MATRIX.md)
[![Protocol](https://img.shields.io/badge/Protocol-3--Round%20Bitwise%20Exact-brightgreen.svg)](reports/V2_FINAL_VERIFICATION.md)

**ForgeTune v2.0** is an ahead-of-time (AOT) C++17 machine learning inference compiler and lightweight native runtime designed for low-latency CPU inference with dynamic batching and zero steady-state native arena allocations after warm-up.

---

## 1. The Engineering Problem

In edge, embedded, and latency-critical CPU inference, executing small-to-medium neural networks through standard deep learning frameworks incurs substantial overhead:
- **Interpreter & Dispatch Overhead:** Dynamic framework runtimes introduce traversal latency, tensor metadata allocation, and operator dispatch costs that often exceed the computation time of the inner linear algebra operations.
- **Fragmented Memory Lifetimes:** Naive per-layer allocation results in recurring heap allocations, cache pollution, and memory fragmentation.
- **Heavyweight Runtimes:** Full runtime dependencies (such as large LLVM-based JIT compilers or extensive C++ framework runtimes) add binary bloat and deployment complexity in constrained environments.

ForgeTune solves this by extracting PyTorch computational graphs into an independent, strongly typed C++17 intermediate representation (IR), applying domain-specific compiler optimizations (graph rewriting, multi-operator loop fusion, and interval-coloring memory planning), and compiling self-contained native shared libraries (`.dll` on Windows, `.so` on Linux) that execute with zero framework dependencies.

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
        F -->|"38.5% - 58.6% Intermediate Buffer Reduction"| G["Static Arena Offsets<br/>(64-byte SIMD Aligned)"]
    end

    subgraph Native["4. Native Codegen & Runtime"]
        G --> H["C++17 Standalone Codegen<br/>(Dynamic Batch Indexing)"]
        H --> I["Zig C++ / Native Toolchain"]
        I --> J["Native Shared Library (.dll on Windows, .so on Linux)<br/>(Dual Versioned C ABI)"]
        J --> K["NativeRuntime (ctypes dispatch)<br/>(Zero Steady-State Native Arena Allocations)"]
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
- **Algebraic Rewriting:** Simplifies identity arithmetic ($x + 0 \to x$ under signed-zero relaxation where $+0.0$ and $-0.0$ are treated as algebraically equivalent) and idempotent activations ($\text{ReLU}(\text{ReLU}(x)) \to \text{ReLU}(x)$). Native C++ kernels compile with strict IEEE 754 precision flags (`/fp:precise` on MSVC) while graph optimizations simplify redundant algebraic operations.
- **Constant Folding & DCE:** Pre-computes weight transformations and removes unreferenced dead subgraphs at compile time.

### Liveness-Based Intermediate Buffer Reuse
- Analyzes activation birth and death intervals across topological execution steps.
- Solves buffer allocation using interval-coloring register-allocation principles.
- Packs all disjoint activation lifetimes into a single pre-allocated `AlignedArenaPool` (SIMD-friendly 64-byte-aligned memory and cache-line boundaries), reducing calculated intermediate activation storage by **38.5% to 58.6%** (note: this metric measures calculated intermediate activation buffer reuse, not total process RAM).

### Dynamic Batch Execution ($B \in [1, B_{max}]$)
- Emits C++ inner loops parameterized on batch length $B$ without requiring recompilation per batch size. Models verify across the default range $B \in [1, 128]$, while the compiler and native runtime support a configurable API maximum up to $1,000,000$.
- Hardened C ABI boundary validation: out-of-bounds requests ($B \le 0$ or $B > B_{max}$) return `-2` and trigger Python `InvalidShapeError` before memory access (with `-1` for null pointers, `-3` for output mismatch, and `-4` for internal runtime exceptions).

### Dual Versioned C ABI
- Defined in `include/forgetune/forgetune_abi.h`:
  - Dynamic Batch V2 Entry Point: `int forgetune_run_v2(const float* inp, size_t inp_len, float* out, size_t out_len, int64_t batch_size)`
  - Legacy Static V1 Entry Point: `int forgetune_run_v1(const float* inp, size_t inp_len, float* out, size_t out_len)` (compatibility/static or inferred-batch execution)
  - Query Exports: `int forgetune_is_dynamic(void)`, `int64_t forgetune_max_batch(void)`

---

## 4. Empirical Benchmarks & Performance Analysis

All benchmarks are measured on AMD64 hardware (Windows 11, PyTorch 2.14.1+cpu). Latency measurements are reported using authoritative on-disk artifacts (`reports/v2_benchmark_report.json` and `reports/v2_paired_b64_benchmark_data.json`).

### Multi-Model Benchmark Table (Dataset: `v2-bench-1793566051-3743367c`)

| Workload Model | Batch Size ($B$) | PyTorch Eager (8 threads) (μs) | PyTorch Eager (1 thread) (μs) | ForgeTune E2E (μs) | **Genuine E2E Speedup (8T)** | **Controlled E2E Speedup (1T)** | ForgeTune Native Kernel (μs) | **Autotune gain vs baseline kernel** | E2E - Kernel Residual (μs) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model A (Dynamic TinyMLP)** | $B=1$ | 35.60 | 35.90 | 9.20 | **3.87x** | **3.90x** | 0.185 | **1.68x** | 9.01 |
| *Linear(16->32) -> ReLU -> Linear(32->8)* | $B=4$ | 43.35 | 38.10 | 9.60 | **4.52x** | **3.97x** | 0.622 | **1.83x** | 8.98 |
| *(Selected: `loop_ikj`)* | $B=16$ | 33.60 | 28.50 | 7.80 | **4.31x** | **3.65x** | 1.734 | **1.96x** | 6.07 |
| | $B=64$ | 51.90 | 49.90 | 21.80 | **2.38x** | **2.29x** | 9.638 | **1.83x** | 12.16 |
| **Model B (Residual MLP)** | $B=1$ | 60.15 | 60.10 | 10.80 | **5.57x** | **5.56x** | 0.090 | **1.11x** | 10.71 |
| *Linear + Residual Skip + ReLU* | $B=4$ | 44.70 | 44.80 | 9.30 | **4.81x** | **4.82x** | 0.353 | **1.10x** | 8.95 |
| *(Selected: `fused_ikj`)* | $B=16$ | 65.80 | 46.00 | 7.00 | **9.40x** | **6.57x** | 1.321 | **1.10x** | 5.68 |
| | $B=64$ | 83.45 | 69.20 | 16.50 | **5.06x** | **4.19x** | 5.232 | **1.10x** | 11.27 |
| **Model C (Small CNN)** | $B=1$ | 110.05 | 90.25 | 11.60 | **9.49x** | **7.78x** | 6.988 | **1.34x** | 4.61 |
| *Conv2d -> ReLU -> Conv2d -> ReLU -> FC* | $B=4$ | 322.80 | 294.40 | 34.70 | **9.30x** | **8.48x** | 24.262 | **1.41x** | 10.44 |
| *(Selected: `tiled_16`)* | $B=16$ | 341.30 | 341.60 | 104.40 | **3.27x** | **3.27x** | 77.403 | **1.48x** | 27.00 |
| | $B=64$ | 394.68 | 451.90 | 385.75 | **1.02x** | **1.17x** | 368.340 | **1.27x** | 17.41 |

*\*Architectural Note on Model B: The autotuner evaluates candidates across the full dynamic batch range ($B \in \{1, 4, 16, 64\}$). Operator fusion (`fused_ikj`) is selected, providing consistent 1.10x–1.11x native kernel speedup over baseline across all batch sizes, and achieving **4.19x controlled E2E speedup over single-threaded PyTorch** ($16.50\ \mu\text{s}$ vs $69.20\ \mu\text{s}$) and **5.06x over 8-threaded PyTorch** ($16.50\ \mu\text{s}$ vs $83.45\ \mu\text{s}$) at $B=64$ by eliminating framework runtime overhead.*

*\*Residual Note: Residuals ($t_{\text{E2E}} - t_{\text{Kernel}}$) represent runtime work outside the inner kernel, including NumPy contiguous buffer conversion, ctypes pointer marshalling, and dynamic shape validation. Because E2E and kernel benchmarks are independently sampled distributions, separate measurement sessions under CPU frequency scaling or timer jitter can produce small signed residuals (as documented in the 30-sample distribution in `reports/V2_BENCHMARK_REPORT.md`). The paired 100-block hypothesis test with alternating execution order provides the definitive controlled statistical verification.*

### Statistically Rigorous Paired Hypothesis Testing ($N=100$ Blocks)

To evaluate Model C at $B=64$ without sequential bias, an interleaved 100-block paired experiment was executed with alternating execution order (`reports/v2_paired_b64_benchmark_data.json`):
- **PyTorch 1-Thread E2E Median:** $353.20\ \mu\text{s}$
- **ForgeTune E2E Median:** $294.90\ \mu\text{s}$
- **Mean Paired Difference ($\bar{d} = t_{PT1} - t_{FT}$):** $+64.14\ \mu\text{s}$ (ForgeTune is significantly faster)
- **Median Difference:** $+54.50\ \mu\text{s}$
- **Standard Error ($SE$):** $5.33\ \mu\text{s}$ ($df = 99$, sample standard deviation $s_d = 53.26\ \mu\text{s}$)
- **One-Sided Non-Inferiority ($H_0: \mu_d \le -17.66\ \mu\text{s}$):** $t = 15.36$, exact Student's t $p = 3.01 \times 10^{-28} \ll 0.0001$. Non-inferiority is decisively certified.
- **Two-Sided TOST Equivalence ($\pm 5\%$ margin, $\pm 17.66\ \mu\text{s}$):** Inconclusive strictly because ForgeTune significantly outperforms PyTorch 1T ($90\%\ \text{CI} = [+55.30, +72.99]\ \mu\text{s}$), placing the entire confidence interval well above $+17.66\ \mu\text{s}$. The test fails equivalence due to genuine performance superiority, not high variance or performance inferiority.
- **Bootstrap 90% Confidence Intervals ($B=10,000$ resamples):**
  - Mean Difference: $[+55.65\ \mu\text{s},\; +73.01\ \mu\text{s}]$
  - Median Difference: $[+52.35\ \mu\text{s},\; +62.10\ \mu\text{s}]$

---

## 5. Quickstart & Installation

### Prerequisites & Verified Platforms
- **Python:** 3.11 or 3.12 (Python 3.11 is supported in package specs `requires-python = ">=3.11, <3.13"` and tested; automated CI workflows specifically target Python 3.12)
- **Windows:** Windows 11 / Windows Server 2022 (AMD64, C++17, Python 3.11/3.12, Zig toolchain, MSVC ABI compatibility)
- **Linux:** Ubuntu 22.04 / 24.04 (x86_64, GCC 12+ / Clang 16+, Python 3.11/3.12, POSIX toolchains)

### Installation & Fresh Checkout Preparation
```bash
# 1. Clone repository
git clone https://github.com/TDheeraj14569/Forgetune.git
cd Forgetune

# 2. Install dependencies and the package in editable mode
pip install -r requirements.txt
pip install -e .

# 3. Build native C++ targets (engine CLI, pybind11 extension, and standalone test suite)
python scripts/build_native.py
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

The repository enforces a defensible, multi-tier automated test suite across Windows and Linux:

```bash
# 1. Run full Python test suite (293 tests)
python -m pytest -v tests/

# 2. Run standalone native C++ unit tests (13 tests)
./bin/forgetune_test_suite   # Linux/POSIX
# or .\bin\forgetune_test_suite.exe on Windows

# 3. Execute master 3-round verification protocol
python scripts/v2_verify_all.py --mode=full
```

### Test Accounting Summary
- **Total Enumerated Tests:** **306 tests** (100% passing, 0 failed, 0 skipped, 0 blocked).
  - **293 Python Tests:** Collected and executed via `pytest`. Covers operator semantics, algebraic simplification, numerical stress, versioned ABI contracts, candidate isolation, and 99 audit defect regressions.
  - **13 Native C++ Tests:** Standalone unit tests executing without Python runtime dependencies in `bin/forgetune_test_suite` / `bin/forgetune_test_suite.exe`.
  - *Note on overlap:* All 13 native C++ test functions are also executed in-process within pytest via `tests/unit/test_pybind11_bindings.py::test_pybind11_native_cpp_test_suite`.
- **Compiler Diagnostics:** Builds with **0 warnings and 0 errors** under `-std=c++17 -O2 -Wall -Wextra` on both Linux and Windows.
  - Upstream Python C API header warnings are selectively silenced using `-Wno-nullability*` flags (`-Wno-nullability`, `-Wno-nullability-completeness`, `-Wno-nullability-completeness-on-arrays`, `-Wno-nullability-declspec`) because external Python headers lack complete nullability annotations on Windows/MinGW toolchains.
  - Intermediate inference compilation applies `-fno-signed-zeros` (fast-math signed-zero relaxation) to treat $+0.0$ and $-0.0$ interchangeably for maximum CPU throughput while preserving IEEE 754 float32 finite ranges. MSVC uses `/fp:precise` to ensure deterministic floating-point semantics.
- **Bitwise Reproducibility:** 3-round verification protocol compares 65,536 float32 values across independent clean compilations with **0 bit divergences** (0/2,097,152 bits).

---

## 7. Scope & Current Limitations

To maintain architectural transparency, the current scope and limitations are explicitly documented:
- **Execution Target:** CPU inference only. No CUDA, Metal, or GPU backends are implemented.
- **Operator Set:** Designed for MLPs and Feed-Forward CNNs. Supports `MATMUL`, `ADD`, `MUL`, `RELU`, `SIGMOID`, `TANH`, `CONV2D`, `FLATTEN`, and fused combinations. Does not support dynamic control flow, RNNs, or Multi-Head Attention mechanisms.
- **Dynamic Dimension Scope:** Dynamic behavior is strictly constrained to the leading batch dimension ($B \le 128$). Internal tensor channel and spatial dimensions ($C, H, W$) are resolved statically at export time.
- **Floating-Point & Numerical Semantics:** Operations use single-precision IEEE 754 float32. Inference kernels use fast-math signed-zero relaxation (`-fno-signed-zeros`), which allows the compiler to treat positive and negative zero as numerically equivalent in SIMD pipelines.
- **Constant Validation:** Graph verification and constant folding strictly reject non-finite float constants (NaN and Infinity) at compile time to ensure generated C++ source code is valid and compiles cleanly without syntax or literal token errors.
- **Threading Model:** Single-threaded execution per runtime instance. Multi-threading is achieved at the application level via concurrent independent runtime instances.

---

## 8. Technology Stack

- **Core Compiler & Engine:** Modern C++17, STL, custom AST & IR DAG.
- **Compilation Toolchains:** Zig C++ (LLVM backend), Clang, GCC, MSVC.
- **Frontend & Graph Capture:** PyTorch 2.14+, `torch.export` (AOT FX graph tracing).
- **Bindings & ABI:** C ABI (`extern "C"`), `ctypes`, `pybind11`.
- **Testing & Verification:** `pytest`, `numpy`, `scipy.stats` (hypothesis testing & bootstrap estimation).
- **CI/CD:** GitHub Actions dual cross-platform matrix (`windows-latest` and `ubuntu-latest`).

---

## 9. Technical Reports & Audit Documentation

For detailed architectural, mathematical, and forensic logs:
- [Requirements Traceability Matrix (REQ-01 to REQ-27)](reports/V2_REQUIREMENTS_MATRIX.md)
- [Benchmark Evaluation & Statistical Analysis](reports/V2_BENCHMARK_REPORT.md)
- [Intermediate-Buffer Storage Reduction Report](reports/V2_MEMORY_REPORT.md)
- [Master Three-Round Verification Report](reports/V2_FINAL_VERIFICATION.md)
- [Authoritative 12-Gate Closure Audit](reports/V2_FINAL_CLOSURE_AUDIT.md)
- [Windows Application Control Forensic Diagnostic](reports/V2_SAC_DIAGNOSTIC.md)
