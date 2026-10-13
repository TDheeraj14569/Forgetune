# ForgeTune IR Specification (Version 1.0)

## 1. Data Types and Numerical Contract

Version 1.0 supports single-precision IEEE-754 floating-point:
- Identifier: `f32` (C++ `float`)
- Storage layout: Contiguous, row-major C-style order
- Rank support: Rank-1 (vectors) and Rank-2 (matrices)

## 2. Operation Kinds

### 2.1 INPUT
- **Arity:** 0 inputs
- **Description:** Declares an external input tensor provided by the caller.
- **Contract:** Fixed rank and static dimensions specialized at compilation time. Non-empty, non-zero dimensions.

### 2.2 CONST
- **Arity:** 0 inputs
- **Description:** Compile-time parameter or constant tensor.
- **Contract:** Buffer size must exactly match `element_count(shape)`.

### 2.3 MATMUL
- **Arity:** 2 inputs: `A`, `B`
- **Semantics:** Matrix multiplication `C = A @ B`
- **Contract:**
  - Rank-2 inputs `A[M, K]` and `B[K, N]`
  - Inner dimension `K` must match exactly
  - Emitted output shape: `[M, N]`

### 2.4 ADD
- **Arity:** 2 inputs: `A`, `B`
- **Semantics:** Elementwise addition with broadcast support
- **Supported Broadcasting Rules:**
  1. Equal shapes: `A[M, N] + B[M, N] -> C[M, N]`
  2. Rank-1 bias vector over rank-2 last dim: `A[M, N] + B[N] -> C[M, N]`
  3. Rank-2 broadcast row: `A[M, N] + B[1, N] -> C[M, N]`
- **Contract:** All other broadcast shapes are explicitly rejected with `InvalidShapeError`.

### 2.5 RELU
- **Arity:** 1 input: `A`
- **Semantics:** Elementwise rectified linear unit: `C = max(A, 0.0)`
- **Contract:** Output shape equals input shape.

### 2.6 FUSED_MATMUL_ADD_RELU
- **Arity:** 3 inputs: `X`, `W`, `B`
- **Semantics:** `C = ReLU(X @ W + B)` evaluated in a single fused loop nest.
- **Contract:**
  - `X: [M, K]`, `W: [K, N]`, `B: [N]` or `[1, N]` or `[M, N]`
  - Output shape: `[M, N]`
  - Intermediate product and sum buffers are never materialized in memory.

### 2.7 OUTPUT
- **Arity:** 1 input
- **Semantics:** Graph-level output marker.

## 3. Serialization Schema

IR graphs are exchanged using JSON with schema version `1.0`:
```json
{
  "schema_version_major": 1,
  "schema_version_minor": 0,
  "inputs": [0],
  "outputs": [5],
  "nodes": [
    {
      "id": 0,
      "op": "INPUT",
      "name": "x",
      "inputs": [],
      "shape": [2, 4],
      "dtype": "f32",
      "const_data": [],
      "deleted": false
    }
  ]
}
```
