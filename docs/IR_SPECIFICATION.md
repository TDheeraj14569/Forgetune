# ForgeTune IR Specification (Version 2.0)

## 1. Data Types and Numerical Contract

ForgeTune v2.0 supports single-precision IEEE-754 floating-point:
- Identifier: `f32` (C++ `float`)
- Storage layout: Contiguous, row-major C-style order (NCHW for 4D activations, OIHW for 4D convolution weights)
- Rank support: Rank-1 (vectors), Rank-2 (matrices), and Rank-4 (tensors)
- Symbolic dimensions: Dynamic leading batch dimension represented by `-1` (`DYNAMIC_DIM`). Remaining non-leading dimensions must be positive static integers. Constants and weights must be fully static.

---

## 2. Operation Kinds

### 2.1 INPUT
- **Arity:** 0 inputs
- **Description:** Declares an external input tensor provided by the caller.
- **Contract:** Rank-2 `[-1, F]` or Rank-4 `[-1, C, H, W]` (or static equivalents). Dynamic dimension is only permitted at index 0.

### 2.2 CONST
- **Arity:** 0 inputs
- **Description:** Compile-time parameter or constant tensor (e.g. weights and biases).
- **Contract:** All dimensions must be strictly positive static integers (`dims[i] > 0`). Buffer size must match `element_count(shape)`.

### 2.3 MATMUL
- **Arity:** 2 inputs: `A`, `B`
- **Semantics:** Matrix multiplication `C = A @ B`
- **Contract:** Rank-2 inputs `A[B, K]` and `B[K, N]`. Inner dimension `K` must match. Output shape: `[B, N]`.

### 2.4 ADD
- **Arity:** 2 inputs: `A`, `B`
- **Semantics:** Elementwise addition with broadcast support
- **Supported Broadcasting Rules:**
  1. Equal shapes: `A[B, ...] + B[B, ...] -> C[B, ...]`
  2. Rank-1 bias over rank-2 last dimension: `A[B, N] + B[N] -> C[B, N]`
  3. Rank-2 broadcast row: `A[B, N] + B[1, N] -> C[B, N]`
  4. Rank-1 channel bias over rank-4 activation: `A[B, C, H, W] + B[C] -> C[B, C, H, W]`

### 2.5 MUL
- **Arity:** 2 inputs: `A`, `B`
- **Semantics:** Elementwise multiplication with broadcast support
- **Supported Broadcasting Rules:**
  1. Equal shapes: `A[B, ...] * B[B, ...] -> C[B, ...]`
  2. Rank-1 vector over rank-2 last dimension: `A[B, N] * B[N] -> C[B, N]`
  3. Rank-2 broadcast row: `A[B, N] * B[1, N] -> C[B, N]`
  4. Rank-1 channel vector over rank-4 activation: `A[B, C, H, W] * B[C] -> C[B, C, H, W]`

### 2.6 RELU
- **Arity:** 1 input: `A`
- **Semantics:** Elementwise rectified linear unit: `C = max(A, 0.0)`
- **Contract:** Output shape equals input shape.

### 2.7 SIGMOID
- **Arity:** 1 input: `A`
- **Semantics:** Numerically stable sigmoid: `C = 1.0 / (1.0 + exp(-A))` with saturation clamping ($A < -88.0 \implies 0.0$, $A > 88.0 \implies 1.0$).
- **Contract:** Output shape equals input shape.

### 2.8 TANH
- **Arity:** 1 input: `A`
- **Semantics:** Hyperbolic tangent: `C = tanh(A)`
- **Contract:** Output shape equals input shape.

### 2.9 CONV2D
- **Arity:** 2 or 3 inputs: `X` (input), `W` (weights), optional `B` (bias)
- **Attributes (`conv_attrs`):** `stride_h`, `stride_w`, `pad_h`, `pad_w`, `dilation_h`, `dilation_w`, `groups` (fixed to 1).
- **Contract:**
  - `X`: Rank-4 `[B, C_in, H_in, W_in]`
  - `W`: Rank-4 `[C_out, C_in, Kh, Kw]`
  - `B` (optional): Rank-1 `[C_out]`
  - Spatial output dimensions:
    - $H_{out} = \lfloor(H_{in} + 2 \times pad_h - (dilation_h \times (Kh - 1) + 1)) / stride_h\rfloor + 1$
    - $W_{out} = \lfloor(W_{in} + 2 \times pad_w - (dilation_w \times (Kw - 1) + 1)) / stride_w\rfloor + 1$
  - Output shape: `[B, C_out, H_out, W_out]`

### 2.10 FLATTEN
- **Arity:** 1 input: `A`
- **Semantics:** Preserves leading batch dimension while flattening spatial dimensions.
- **Contract:**
  - Rank-4 input `[B, C, H, W]` flattens to Rank-2 `[B, C * H * W]`.
  - Rank-2 input `[B, N]` remains `[B, N]`.
  - Rank-1 or Rank-3 inputs are rejected.

### 2.11 FUSED_CONV2D_RELU
- **Arity:** 2 or 3 inputs: `X`, `W`, optional `B`
- **Semantics:** Fuses 2D convolution and ReLU into a single loop nest without materializing the unactivated convolution output.
- **Contract:** Same input shapes and attributes as `CONV2D`.

### 2.12 FUSED_MATMUL_ADD_RELU
- **Arity:** 3 inputs: `X`, `W`, `B`
- **Semantics:** `C = ReLU(X @ W + B)` evaluated in a single fused loop nest.

### 2.13 OUTPUT
- **Arity:** 1 input
- **Semantics:** Graph-level output marker.

---

## 3. Serialization Schema 2.0

IR graphs are exchanged using JSON with schema version `2.0`:
```json
{
  "schema_version_major": 2,
  "schema_version_minor": 0,
  "inputs": [6],
  "outputs": [10],
  "nodes": [
    {
      "id": 6,
      "op": "INPUT",
      "name": "x",
      "inputs": [],
      "shape": [-1, 1, 8, 8],
      "dtype": "f32",
      "const_data": [],
      "deleted": false
    },
    {
      "id": 7,
      "op": "CONV2D",
      "name": "conv1",
      "inputs": [6, 0, 1],
      "shape": [-1, 8, 8, 8],
      "dtype": "f32",
      "const_data": [],
      "deleted": false,
      "conv_attrs": {
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 1,
        "pad_w": 1,
        "dilation_h": 1,
        "dilation_w": 1,
        "groups": 1
      }
    }
  ]
}
```
