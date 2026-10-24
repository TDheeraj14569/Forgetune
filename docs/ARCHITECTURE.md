# ForgeTune v2.0 Architecture Specification

## 1. System Overview

ForgeTune v2.0 is a specification-driven, correctness-verified, self-optimizing machine-learning compiler built from scratch for CPU inference. It translates PyTorch computation graphs into custom C++ Intermediate Representation (IR Schema 2.0), verifies graph structure, type safety, and dynamic shape contracts, optimizes through an extensible pass manager (including constant folding, algebraic simplification, dead-code elimination, and multi-operator fusion), performs compile-time liveness interval analysis to reuse intermediate memory buffers, generates standalone C++ CPU kernels supporting dynamic batch execution ($B \in [1, 128]$), compiles them into shared libraries, and autotunes kernel candidates through an empirical correctness gate on the host machine.

```
+-----------------------------------------------------------------------------------------+
|                                    Frontend (Python)                                    |
|  PyTorch Model  --->  torch.export.export  --->  Dynamic Graph Signature & Lifted Weights|
+-----------------------------------------------------------------------------------------+
                                             |
                                    Normalized Graph JSON (Schema 2.0)
                                             v
+-----------------------------------------------------------------------------------------+
|                                 Compiler Engine (C++17)                                 |
|  IR Graph Parser  --->  Graph Verifier  --->  Pass Manager  --->  Memory Planner       |
|                              |                   |                        |             |
|                      Dynamic Dim Bounds    Constant Folding       Liveness Intervals    |
|                      Rank-2 & Rank-4       Alg Simplification     Interval Coloring     |
|                      Acyclic & Types       Dead-Code Elimination  64B Aligned Arena     |
|                                            Operator Fusion                              |
|                                            - MatMul+Add+ReLU                            |
|                                            - Conv2D+ReLU                                |
+-----------------------------------------------------------------------------------------+
                                             |
                                  Optimized IR + Memory Plan
                                             v
+-----------------------------------------------------------------------------------------+
|                                  Code Generator (C++17)                                 |
|         Emits Standalone C++ Source with Dynamic Batch Indexing & Arena Offsets         |
|         Dual C ABI: forgetune_run (5-arg) + forgetune_run_v1 (4-arg) + query exports    |
+-----------------------------------------------------------------------------------------+
                                             |
                                   Standalone C++ Source (.cpp)
                                             v
+-----------------------------------------------------------------------------------------+
|                                  Native Toolchain (C++)                                 |
|                     Zig C++ / Clang++ / GCC / MSVC  --->  Shared Library (.dll / .so)   |
+-----------------------------------------------------------------------------------------+
                                             |
                                 ctypes Native Runtime Loader
                                             v
+-----------------------------------------------------------------------------------------+
|                                  Autotuning & Runtime                                   |
|   Dynamic Batch Validation  --->  Correctness Gate  --->  Microbenchmarking             |
|   (B in [1, 128])                 (Ref & PyTorch)         (Pure C++ steady_clock)       |
+-----------------------------------------------------------------------------------------+
```

## 2. Subsystem Separation of Responsibilities

1. **Python Frontend (`forgetune.export_frontend`)**:
   - Uses `torch.export.export` with `torch.export.Dim("batch", min=1, max=128)` to extract computation graphs ahead-of-time.
   - Lifts parameters and buffers; normalizes ATen operators (`aten.linear`, `aten.conv2d`/`aten.convolution`, `aten.relu`, `aten.sigmoid`, `aten.tanh`, `aten.flatten`, `aten.view`/`aten.reshape`).
   - Automatically transposes linear weights from PyTorch layout `[N, K]` to canonical row-major layout `[K, N]`.
   - Serializes to versioned IR Schema 2.0 with symbolic leading dimension `-1`.

2. **C++ IR & Graph Verifier (`forgetune::IRGraph`, `forgetune::verify_graph`)**:
   - Implements source-of-truth IR representation with stable node IDs and dynamic tensor shapes.
   - Enforces strict verification: dynamic dimension only permitted at leading batch index (dim 0); weights and constants must be fully static; convolution attributes (stride, pad, dilation, groups) and spatial output shapes must be valid.

3. **Pass Manager & Operator Fusion (`forgetune::PassManager`, `forgetune::fuse_conv2d_relu`)**:
   - Preserves numerical and graph semantics.
   - `ConstantFolding`: Evaluates static subgraphs at compile time.
   - `AlgebraicSimplification`: Rewrites additive zeros, multiplication identities, and redundant activations.
   - `DeadCodeElimination`: Prunes unused nodes while preserving declared graph inputs and outputs.
   - `OperatorFusion`: Detects `MATMUL + ADD + RELU` and `CONV2D + RELU` single-consumer subgraphs, transforming them into fused kernels.

4. **Liveness-Based Memory Planner (`forgetune::MemoryPlanner`)**:
   - Performs topological liveness interval analysis $[birth, death]$ for all intermediate activation buffers.
   - Employs greedy interval-coloring to assign non-overlapping buffers to shared offsets within a single 64-byte aligned memory arena.
   - Reduces intermediate buffer memory by 42.9% to 60.0% while eliminating stack overflow hazards on Windows.

5. **Code Generator & Dual ABI (`forgetune::CodeGenerator`)**:
   - Emits clean, readable C++17 source code without third-party tensor libraries.
   - Manages dynamic batch indexing: outer loops iterate over runtime batch size $B$.
   - Intermediate buffers are mapped into a single `alignas(64) static thread_local std::vector<float> arena_storage`.
   - Dual C ABI export:
     - `forgetune_run`: 5 arguments for dynamic graphs `(const float* inp, size_t inp_count, float* out, size_t out_count, int64_t batch_size)`.
     - `forgetune_run_v1`: 4 arguments for backward compatibility.
     - `forgetune_is_dynamic`: returns 1 for dynamic models, 0 for static.
     - `forgetune_max_batch`: returns configured maximum batch (default 128).
     - `forgetune_benchmark_kernel`: microbenchmark kernel running entirely inside pure C++ via `std::chrono::steady_clock`.

6. **Native Runtime & Loader (`forgetune.runtime.NativeRuntime`)**:
   - Loads compiled `.dll` directly with security policy compliance (no evasion, no stream deletion, no temporary staging). Bounded retry is strictly reserved for transient file-sharing locks.
   - Introspects exported symbols to determine dynamic capabilities.
   - Enforces contract validations: validates input rank, static dimensions, dynamic batch bounds ($1 \le B \le 128$), ensures contiguous float32 data, and verifies native return status codes.
