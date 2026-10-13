# ForgeTune Architecture Specification

## 1. System Overview

ForgeTune is a specification-driven, correctness-checked, self-optimizing machine-learning compiler built from scratch. It translates PyTorch computation graphs into custom C++ Intermediate Representation (IR), verifies graph structure and type safety, optimizes through an extensible pass manager, generates standalone C++ CPU kernels, and autotunes kernel candidates through an empirical correctness gate.

```
+-----------------------------------------------------------------------------------------+
|                                    Frontend (Python)                                    |
|  PyTorch Model  --->  torch.export.export  --->  Graph Signature & Weight Extraction   |
+-----------------------------------------------------------------------------------------+
                                             |
                                    Normalized Graph JSON
                                             v
+-----------------------------------------------------------------------------------------+
|                                 Compiler Engine (C++17)                                 |
|  IR Graph Parser  --->  Graph Verifier  --->  Pass Manager  --->  Code Generator        |
|                              |                   |                     |                |
|                      Acyclic & Typing     Constant Folding     Kernel Variants:         |
|                         Contracts            Alg Simplification  - Naive i,j,k          |
|                                                  DCE             - Cache-Friendly i,k,j |
|                                             Operator Fusion      - 2D Tiled             |
|                                                                  - Fused Loop Nest      |
+-----------------------------------------------------------------------------------------+
                                             |
                                  Standalone C++ Source (.cpp)
                                             v
+-----------------------------------------------------------------------------------------+
|                                  Native Toolchain (C++)                                 |
|                     Clang++ / GCC / MSVC / Zig C++  --->  Shared Library (.dll / .so)   |
+-----------------------------------------------------------------------------------------+
                                             |
                                       ctypes Binding
                                             v
+-----------------------------------------------------------------------------------------+
|                                  Autotuning & Runtime                                   |
|   Correctness Gate (PyTorch & Ref Interp)  --->  Host Benchmarking  --->  Manifest      |
+-----------------------------------------------------------------------------------------+
```

## 2. Subsystem Separation of Responsibilities

1. **Python Frontend (`forgetune.export_frontend`)**:
   - Uses `torch.export` to extract computation graphs.
   - Normalizes ATen operations (e.g. `aten.linear` to canonical `MATMUL` + `ADD`, transposing linear weight tensors from `[N, K]` to `[K, N]`).
   - Exports versioned JSON specification.

2. **C++ IR & Compiler Engine (`forgetune::Compiler`)**:
   - Implements source-of-truth IR graph representation with stable IDs and use counts.
   - Enforces strict contract verification (`verify_graph`).
   - Executes optimization passes (`ConstantFolding`, `AlgebraicSimplification`, `DeadCodeElimination`, `OperatorFusion`).
   - Emits standalone C++ source without external ML or tensor library dependencies.

3. **Native Compiler Driver (`forgetune.native_compiler`)**:
   - Discovers native toolchains (`zig c++`, `clang++`, `g++`, `cl.exe`).
   - Executes compilation with shell disabled and timeout protection.

4. **Runtime Loader (`forgetune.runtime`)**:
   - Loads compiled shared library via `ctypes`.
   - Binds `extern "C" int forgetune_run(const float* input, size_t input_count, float* output, size_t output_count)`.
   - Validates shapes, dtypes, null pointers, and length contracts before invocation.

5. **Autotuning & Benchmarking (`forgetune.autotuner`, `forgetune.benchmark`)**:
   - Explores candidate search space (variants, tile sizes, compiler flags).
   - Gates every candidate against reference interpreter and PyTorch outputs across multiple randomized inputs.
   - Measures steady-state execution time using monotonic high-resolution timers (`time.perf_counter_ns`).
   - Selects optimal candidate and stores cached results with atomic file replacement.
