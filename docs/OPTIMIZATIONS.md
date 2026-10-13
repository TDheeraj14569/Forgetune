# ForgeTune Optimization Pass Guide

ForgeTune incorporates a modular optimization pass framework managed by `PassManager`. Every transformation enforces strict pre-conditions, verifies the graph post-transformation, and guarantees numerical semantics.

## 1. Pass A: Constant Folding (`ConstantFoldingPass`)

- **Precondition:** An operation node whose operands are all `CONST` nodes.
- **Transformation:** Computes the operation result ahead of time using the reference evaluation engine and replaces the node with a single `CONST` node containing the computed values.
- **Verification:** Ensures the output tensor shape and element count match the folded buffer size.

## 2. Pass B: Algebraic Simplification (`AlgebraicSimplificationPass`)

- **Rule 1 (Additive Identity):** `ADD(X, 0)` -> `X` when `0` is an all-zero tensor with matching shape.
- **Rule 2 (ReLU Idempotence):** `RELU(RELU(X))` -> `RELU(X)`.
- **Precondition:** Operands must satisfy floating-point equivalence without altering NaN or signed zero contracts.

## 3. Pass C: Dead-Code Elimination (`DeadCodeEliminationPass`)

- **Precondition:** Nodes in the graph that do not contribute to any declared graph output.
- **Transformation:** Performs backwards reachability traversal from all nodes in `graph.get_outputs()`. Any unreachable node is soft-deleted.
- **Safety Contract:** Never deletes nodes that are still consumed by live downstream operations.

## 4. Pass D: Operator Fusion (`OperatorFusionPass`)

- **Pattern:** `MATMUL(%X, %W) -> ADD(%M, %B) -> RELU(%A)`
- **Legality Checks:**
  1. `%M` (MATMUL) must have use count exactly equal to 1 (consumed only by `%A`), and `%M` must not be a graph output.
  2. `%A` (ADD) must have use count exactly equal to 1 (consumed only by `%R`), and `%A` must not be a graph output.
- **Transformation:**
  - Replaces `%R` with `%F = FUSED_MATMUL_ADD_RELU(%X, %W, %B)`.
  - Replaces all uses of `%R` with `%F`.
  - Deletes intermediate nodes `%M`, `%A`, and `%R`.
- **Benefit:** Eliminates materialization of intermediate matrix multiplication and addition memory buffers, maximizing L1 cache locality and eliminating store-load round trips.
