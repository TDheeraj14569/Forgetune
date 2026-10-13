# ForgeTune v2.0 Operator Specification

This document provides mathematical definitions, broadcasting rules, and implementation details for every supported operator in ForgeTune v2.0.

---

## 1. Matrix Multiplication (`MATMUL`)

- **Formula:** $C_{i, j} = \sum_{k=0}^{K-1} A_{i, k} \cdot B_{k, j}$
- **Inputs:**
  - $A$: Float32 tensor of shape $[B, K]$ (where $B$ is dynamic or static)
  - $B$: Float32 tensor of shape $[K, N]$ (transposed weight matrix)
- **Output:** Float32 tensor of shape $[B, N]$
- **Kernel Variants:**
  - `naive_ijk`: Canonical 3-nested loop order ($i \to j \to k$).
  - `loop_ikj`: Cache-friendly loop order ($i \to k \to j$) with contiguous memory writes.
  - `tiled`: 2D/3D cache-blocked matrix multiplication with tile sizes $T_m, T_n, T_k \in \{8, 16\}$.
- **Dynamic Batching:** Loop bound for dimension $i$ dynamically evaluates up to runtime batch $B$.

---

## 2. Elementwise Addition (`ADD`)

- **Formula:** $C = A + B$
- **Supported Broadcasting Scenarios:**
  1. **Strictly Identical Shapes:** $A[B, ...] + B[B, ...]$
  2. **1D Bias Broadcasting over Rank-2:** $A[B, N] + B[N] \implies C_{i, j} = A_{i, j} + B_j$
  3. **Row Broadcasting over Rank-2:** $A[B, N] + B[1, N] \implies C_{i, j} = A_{i, j} + B_{0, j}$
  4. **Channel Bias over Rank-4 Activation:** $A[B, C, H, W] + B[C] \implies C_{b, c, h, w} = A_{b, c, h, w} + B_c$

---

## 3. Elementwise Multiplication (`MUL`)

- **Formula:** $C = A \odot B$
- **Supported Broadcasting Scenarios:**
  1. **Identical Shapes:** $A[B, ...] \odot B[B, ...]$
  2. **Vector Scaling over Rank-2:** $A[B, N] \odot B[N] \implies C_{i, j} = A_{i, j} \cdot B_j$
  3. **Row Scaling over Rank-2:** $A[B, N] \odot B[1, N] \implies C_{i, j} = A_{i, j} \cdot B_{0, j}$
  4. **Channel Scaling over Rank-4:** $A[B, C, H, W] \odot B[C] \implies C_{b, c, h, w} = A_{b, c, h, w} \cdot B_c$

---

## 4. Rectified Linear Unit (`RELU`)

- **Formula:** $C = \max(0.0f, A)$
- **Output Shape:** Identical to input shape.
- **Optimization:** Pattern-matched into `FUSED_MATMUL_ADD_RELU` or `FUSED_CONV2D_RELU`.

---

## 5. Sigmoid Activation (`SIGMOID`)

- **Formula:** $\sigma(x) = \frac{1}{1 + e^{-x}}$
- **Numerical Stability Safeguards:**
  - When $x < -88.0f$, $e^{-x}$ overflows 32-bit float; output clamped to $0.0f$.
  - When $x > 88.0f$, $e^{-x} \approx 0.0f$; output clamped to $1.0f$.
- **Output Shape:** Identical to input shape.

---

## 6. Hyperbolic Tangent (`TANH`)

- **Formula:** $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
- **Implementation:** Evaluated via C++ `<cmath>` `std::tanh` matching IEEE-754 precision.
- **Output Shape:** Identical to input shape.

---

## 7. 2D Convolution (`CONV2D`)

- **Formula:**
  $$Y_{b, c_{out}, h_{out}, w_{out}} = B_{c_{out}} + \sum_{c_{in}=0}^{C_{in}-1} \sum_{kh=0}^{Kh-1} \sum_{kw=0}^{Kw-1} X_{b, c_{in}, h_{in}, w_{in}} \cdot W_{c_{out}, c_{in}, kh, kw}$$
  where:
  $$h_{in} = h_{out} \cdot stride_h - pad_h + kh \cdot dilation_h$$
  $$w_{in} = w_{out} \cdot stride_w - pad_w + kw \cdot dilation_w$$
- **Layout:** Activations are NCHW ($[B, C, H, W]$); weights are OIHW ($[C_{out}, C_{in}, Kh, Kw]$).
- **Attributes:**
  - `stride_h`, `stride_w`: Convolution step size (default 1).
  - `pad_h`, `pad_w`: Zero-padding added to input boundaries (default 0).
  - `dilation_h`, `dilation_w`: Spacing between kernel elements (default 1).
  - `groups`: Channel grouping (must equal 1).
- **Output Dimension Calculation:**
  $$H_{out} = \left\lfloor \frac{H_{in} + 2 \cdot pad_h - (dilation_h \cdot (Kh - 1) + 1)}{stride_h} \right\rfloor + 1$$
  $$W_{out} = \left\lfloor \frac{W_{in} + 2 \cdot pad_w - (dilation_w \cdot (Kw - 1) + 1)}{stride_w} \right\rfloor + 1$$

---

## 8. Spatial Flattening (`FLATTEN`)

- **Semantics:** Reshapes a multi-dimensional activation tensor into a 2D matrix while strictly preserving the leading batch dimension.
- **Contract:**
  - Rank-4 input $[B, C, H, W]$ flattens to Rank-2 $[B, C \cdot H \cdot W]$.
  - Zero-copy / pointer reuse in intermediate memory arena.

---

## 9. Fused Convolution + ReLU (`FUSED_CONV2D_RELU`)

- **Semantics:** Evaluates direct 2D convolution and applies $\max(0.0f, x)$ in-register before writing to memory.
- **Memory Optimization:** Eliminates intermediate convolution activation tensor memory allocation and round-trip cache transfers.
