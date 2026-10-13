# FORGETUNE V2.0 — INITIAL ARCHITECTURAL AUDIT AND BASELINE ASSESSMENT

**Project:** ForgeTune Self-Optimizing CPU Machine-Learning Compiler  
**Audit Phase:** Milestone 0 (Pre-Implementation Baseline Discovery)  
**Date:** October 14, 2026  
**Auditor:** Senior Compiler Systems & Machine Learning Engineer  
**Status:** **BASELINE ESTABLISHED — READY FOR V2.0 DEVELOPMENT**

---

## 1. Executive Summary

This initial audit establishes the architectural baseline of ForgeTune v1.0 and defines the concrete engineering plan for **ForgeTune v2.0**.

ForgeTune v1.0 successfully implemented an ahead-of-time inference compilation pipeline for static 2D MLP models using `torch.export`, canonical IR representation, a reference interpreter, basic optimization passes (constant folding, algebraic simplification, dead-code elimination, and operator fusion), standalone C++17 code generation with 5 kernel variants, a C ABI runtime interface, and a correctness-gated autotuner. All 127 baseline tests (8 native C++, 44 baseline Python, and 75 custom acceptance tests) pass with exit code 0.

However, v1.0 has strict foundational limitations:
1. **Fully Static Shapes Only:** Tensors must have statically fixed shapes; batch dimension $B$ cannot vary at runtime.
2. **Rank-2 Only:** Activations are restricted to 2D matrices ($[M, K]$ and $[M, N]$); rank-4 spatial tensors ($[B, C, H, W]$) are unsupported.
3. **Restricted Operator Set:** Only `INPUT`, `CONST`, `MATMUL`, `ADD`, `RELU`, `OUTPUT`, and `FUSED_MATMUL_ADD_RELU` are supported.
4. **No Convolutional Support:** Computer vision workloads cannot be compiled due to missing `CONV2D` and `FLATTEN`.
5. **No Intermediate Buffer Reuse:** Intermediate buffers are allocated per-node, resulting in unnecessary memory overhead.
6. **Fixed ABI Contract:** The native runtime interface requires exact static element counts (`input_count == total_elements`).

ForgeTune v2.0 addresses each of these limitations, transforming the compiler into a dynamic-batch CPU inference compiler supporting rank-2 and rank-4 activations, elementwise math (`MUL`, `SIGMOID`, `TANH`), 2D convolution (`CONV2D`), tensor flattening (`FLATTEN`), liveness-based buffer reuse, and end-to-end execution of Dynamic TinyMLP, Residual MLP, and Convolutional Neural Networks (CNN).

---

## 2. Environment & Toolchain Discovery

The current host development environment was inspected:

| Component | Detected Specification | Verification Status |
|---|---|---|
| **Operating System** | Windows 11 Pro AMD64 (Build 26100) | Verified |
| **Processor** | AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD | Verified |
| **Python** | Python 3.12.14 (`.venv\Scripts\python.exe`) | Verified |
| **PyTorch** | PyTorch 2.14.1+cpu (`torch.export`, `torch.fx`) | Verified |
| **NumPy** | NumPy 2.5.2 | Verified |
| **pybind11** | pybind11 3.1.0 | Verified |
| **Native Compiler** | Zig C++ Compiler 0.16.0 (`-target x86_64-windows-gnu`, C++17) | Verified |
| **Test Runner** | pytest 9.1.1 | Verified |
| **Git Branch** | `v2.0-dev` (branched from tagged `v1.0.0` commit `a07e807`) | Verified |

---

## 3. Current v1.0 Architecture Analysis

### 3.1 Intermediate Representation (`include/forgetune/ir.hpp`, `src/ir.cpp`)
- `TensorShape`: Stores `std::vector<int64_t> dims`. In v1.0, all dimensions are positive integers (`d > 0`). Rank is strictly checked.
- `IRNode`: Stores `id`, `op`, `name`, `inputs`, `output_meta`, `const_data`, and `deleted` flag.
- `IRGraph`: Directed acyclic graph with `nodes_`, `inputs_`, and `outputs_`. Enforces topological ordering and acyclicity.
- Schema version: `IR_SCHEMA_VERSION_MAJOR = 1, IR_SCHEMA_VERSION_MINOR = 0`.

### 3.2 Graph Verifier (`include/forgetune/graph_verifier.hpp`, `src/graph_verifier.cpp`)
- Enforces non-empty declared inputs and outputs.
- Verifies acyclicity via topological sort.
- Validates arity: `INPUT` (0), `CONST` (0), `MATMUL` (2), `ADD` (2), `RELU` (1), `OUTPUT` (1), `FUSED_MATMUL_ADD_RELU` (3).
- Verifies rank-2 shapes for `MATMUL` ($[M, K] \times [K, N] \to [M, N]$).
- Verifies broadcast rules for `ADD`: exact shape matching, rank-1 vector bias ($[M, N] + [N] \to [M, N]$), or rank-2 broadcast row ($[M, N] + [1, N] \to [M, N]$).
- Rejects invalid ranks, negative IDs, and deleted nodes.

### 3.3 Reference Interpreter (`include/forgetune/interpreter.hpp`, `src/interpreter.cpp`)
- Evaluates static float32 tensors in topological order.
- Supports `INPUT`, `CONST`, `MATMUL`, `ADD`, `RELU`, `OUTPUT`, and `FUSED_MATMUL_ADD_RELU`.
- Serves as the ground truth oracle for compiler validation and autotuner correctness gating.

### 3.4 Optimization Pipeline (`src/pass_manager.cpp`, `src/optimizer.cpp`, `src/fusion.cpp`)
- `ConstantFoldingPass`: Folds compile-time static operations whose operands are all `CONST`.
- `AlgebraicSimplificationPass`: Eliminates $X + 0 \to X$, $0 + X \to X$, and $\text{ReLU}(\text{ReLU}(X)) \to \text{ReLU}(X)$.
- `DeadCodeEliminationPass`: Computes backwards reachability from outputs and preserves declared inputs as live ABI roots.
- `OperatorFusionPass`: Detects pattern $\text{ReLU}(\text{Add}(\text{MatMul}(X, W), B))$ and fuses into `FUSED_MATMUL_ADD_RELU`, strictly gating on use count == 1 and non-output status.

### 3.5 Code Generator & Runtime ABI (`src/codegen.cpp`, `python/forgetune/runtime.py`)
- Emits C++17 standalone kernel with compile-time static constants.
- Emits specialized kernels: `naive_ijk`, `loop_ikj`, `tiled`, `fused_ikj`, `fused_tiled`.
- Exports C ABI entry points:
  - `int forgetune_run(const float* input, size_t input_count, float* output, size_t output_count) noexcept`
  - `double forgetune_benchmark_kernel(...) noexcept`
- Runtime checks pointers (`nullptr` $\to -1$), input count ($-2$), and output count ($-3$).

---

## 4. Current Baseline Verification Results

The existing verification procedures were executed to establish the baseline before any v2.0 code changes:

| Verification Suite | Target Executable / Command | Test Count | Result |
|---|---|:---:|:---:|
| **C++ Native Test Suite** | `bin/forgetune_test_suite.exe` | 8 | **8 / 8 PASSED** |
| **Python PyTest Suite** | `pytest tests/` | 119 | **119 / 119 PASSED** |
| **Acceptance Suite** | `python scripts/acceptance_test.py` | 75 | **75 / 75 PASSED** |
| **TinyMLP E2E Execution** | `python examples/tiny_mlp.py` | 1 | **PASSED** ($\Delta = 2.98 \times 10^{-8}$) |

**Baseline Integrity:** The starting repository is 100% clean and passing with zero pre-existing failures.

---

## 5. Architectural Gap Analysis: v1.0 vs v2.0

| Feature | ForgeTune v1.0 Baseline | ForgeTune v2.0 Target |
|---|---|---|
| **Batch Dimension** | Static only ($M$ fixed at compile time) | **Dynamic leading batch $B \in [1, B_{max}]$** |
| **Maximum Batch** | N/A (compile-time fixed) | **Configurable $B_{max}$ (default 128)** |
| **Tensor Ranks** | Rank-2 only ($[M, K]$) | **Rank-2 ($[B, F]$) and Rank-4 ($[B, C, H, W]$)** |
| **Operator Vocabulary** | 7 ops (Input, Const, MatMul, Add, ReLU, Output, Fused) | **12 ops (+ MUL, SIGMOID, TANH, CONV2D, FLATTEN)** |
| **Convolution** | Unsupported | **Float32 2D Convolution (NCHW, OIHW, stride, pad, dil)** |
| **Tensor Flattening** | Unsupported | **Batch-preserving spatial flatten ($[B, C, H, W] \to [B, C\cdot H\cdot W]$)** |
| **Elementwise Math** | Add, ReLU only | **+ Elementwise Multiply, Sigmoid, Tanh** |
| **Model Workloads** | TinyMLP only | **Dynamic TinyMLP, Residual MLP, Small CNN** |
| **Memory Planning** | Static per-node allocation | **Liveness-based intermediate buffer reuse** |
| **Runtime ABI** | Static element counts | **Dynamic batch ABI with shape & buffer validation** |
| **IR Schema Version** | Version 1.0 | **Version 2.0 (symbolic dimension support)** |

---

## 6. ForgeTune v2.0 Implementation Roadmap

The implementation will proceed through the required 8 milestone gates:

1. **Milestone 1 — IR and Shape System:**
   - Update `TensorShape` to support symbolic dimension (`-1` or `DYNAMIC_DIM = -1`).
   - Add rank-4 support and validation.
   - Bump schema version to 2.0.
   - Update graph verifier for dynamic batch compatibility and new ops.
2. **Milestone 2 — Reference Interpreter & Operator Semantics:**
   - Implement `MUL`, `SIGMOID`, `TANH`, `CONV2D`, `FLATTEN` in C++ interpreter.
   - Add unit and differential tests for each operator.
3. **Milestone 3 — Actual PyTorch Export Frontend:**
   - Update `export_frontend.py` to use `torch.export.Dim("batch", min=1, max=128)`.
   - Normalize `aten.convolution`, `aten.mul`, `aten.sigmoid`, `aten.tanh`, `aten.flatten`.
4. **Milestone 4 — Optimizations & Memory Planning:**
   - Extend constant folding, algebraic simplification, and DCE for v2.0 ops.
   - Add Conv2D + ReLU fusion pass.
   - Implement intermediate buffer reuse based on tensor liveness analysis.
5. **Milestone 5 — Native Code Generation & Runtime:**
   - Generate C++ code for dynamic batch indexing, Conv2D, Mul, Sigmoid, Tanh, Flatten.
   - Extend runtime ABI to accept dynamic batch size $B$ and validate buffer sizes.
   - Update `NativeRuntime` loader.
6. **Milestone 6 — Required Model Workloads:**
   - Implement and verify Model A (Dynamic TinyMLP), Model B (Residual MLP), Model C (Small CNN).
7. **Milestone 7 — Autotuning & Performance:**
   - Extend candidate space and autotuner for convolution and dynamic batch.
   - Measure isolated kernel latency vs end-to-end Python latency.
8. **Milestone 8 — Documentation & Release Verification:**
   - Comprehensive documentation, 3 verification rounds, and 3 independent audits.
