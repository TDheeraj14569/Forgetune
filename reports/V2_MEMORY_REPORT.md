# FORGETUNE V2.0 — CALCULATED INTERMEDIATE-BUFFER STORAGE REDUCTION REPORT

**Date:** 2026-10-15 03:07:21
**Optimization Technique:** Interval-coloring liveness analysis with 64-byte aligned arena reuse
**Configured Maximum Dynamic Batch Size ($B_{max}$):** 128

---

## 1. Executive Summary

In naive neural network execution, every intermediate activation tensor is assigned dedicated storage.
ForgeTune v2.0 performs compile-time liveness interval analysis to determine non-overlapping activation lifetimes,
packing intermediate buffers into a single 64-byte cache-line aligned `thread_local` memory arena.
Across all three benchmark models, ForgeTune v2.0 achieves between **42.9% and 60.0% calculated intermediate-buffer storage reduction**.

---

## 2. Memory Reduction Overview Table

| Model Name | Logical Tensors Reused | Physical Offset Slots | Unoptimized Naive Memory | Optimized Arena Memory | Calculated Intermediate-Buffer Storage Reduction (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Model A (Dynamic TinyMLP)** | 5 tensors | 2 physical slots | 57,344 B (56.0 KB) | 32,768 B (32.0 KB) | **42.9%** |
| **Model B (Residual MLP)** | 9 tensors | 4 physical slots | 61,440 B (60.0 KB) | 24,576 B (24.0 KB) | **60.0%** |
| **Model C (Small CNN)** | 7 tensors | 3 physical slots | 927,744 B (906.0 KB) | 524,288 B (512.0 KB) | **43.5%** |

---

## 3. Detailed Per-Model Allocation Schemes

### Model A (Dynamic TinyMLP)
- **Logical Intermediate Tensors:** 5
- **Physical Reusable Arena Slots:** 2
- **Unoptimized Static Allocation Total:** 57,344 bytes
- **Optimized Reused Arena Size:** 32,768 bytes
- **Net Savings:** 24,576 bytes (42.9%)

| Node ID | Liveness Interval [Birth, Death] | Buffer Size (Bytes) | Arena Offset (64B Aligned) | Reused Spatial Slot |
|:---:|:---:|:---:|:---:|:---:|
| Node  8 | Step [ 8, 10] |  16,384 B |       0 B | `Slot @ 0x00000` |
| Node 11 | Step [11, 12] |   4,096 B |       0 B | `Slot @ 0x00000` |
| Node  6 | Step [ 6,  7] |  16,384 B |       0 B | `Slot @ 0x00000` |
| Node 10 | Step [10, 11] |   4,096 B |  16,384 B | `Slot @ 0x04000` |
| Node  7 | Step [ 7,  8] |  16,384 B |  16,384 B | `Slot @ 0x04000` |

### Model B (Residual MLP)
- **Logical Intermediate Tensors:** 9
- **Physical Reusable Arena Slots:** 4
- **Unoptimized Static Allocation Total:** 61,440 bytes
- **Optimized Reused Arena Size:** 24,576 bytes
- **Net Savings:** 36,864 bytes (60.0%)

| Node ID | Liveness Interval [Birth, Death] | Buffer Size (Bytes) | Arena Offset (64B Aligned) | Reused Spatial Slot |
|:---:|:---:|:---:|:---:|:---:|
| Node 17 | Step [17, 18] |   2,048 B |       0 B | `Slot @ 0x00000` |
| Node 10 | Step [10, 15] |   8,192 B |       0 B | `Slot @ 0x00000` |
| Node  8 | Step [ 8,  9] |   8,192 B |       0 B | `Slot @ 0x00000` |
| Node 18 | Step [18, 19] |   2,048 B |   2,048 B | `Slot @ 0x00800` |
| Node 14 | Step [14, 15] |   8,192 B |   8,192 B | `Slot @ 0x02000` |
| Node 12 | Step [12, 13] |   8,192 B |   8,192 B | `Slot @ 0x02000` |
| Node  9 | Step [ 9, 10] |   8,192 B |   8,192 B | `Slot @ 0x02000` |
| Node 15 | Step [15, 17] |   8,192 B |  16,384 B | `Slot @ 0x04000` |
| Node 13 | Step [13, 14] |   8,192 B |  16,384 B | `Slot @ 0x04000` |

### Model C (Small CNN)
- **Logical Intermediate Tensors:** 7
- **Physical Reusable Arena Slots:** 3
- **Unoptimized Static Allocation Total:** 927,744 bytes
- **Optimized Reused Arena Size:** 524,288 bytes
- **Net Savings:** 403,456 bytes (43.5%)

| Node ID | Liveness Interval [Birth, Death] | Buffer Size (Bytes) | Arena Offset (64B Aligned) | Reused Spatial Slot |
|:---:|:---:|:---:|:---:|:---:|
| Node 14 | Step [14, 15] |   5,120 B |       0 B | `Slot @ 0x00000` |
| Node 11 | Step [11, 13] | 131,072 B |       0 B | `Slot @ 0x00000` |
| Node  9 | Step [ 9, 10] | 131,072 B |       0 B | `Slot @ 0x00000` |
| Node  7 | Step [ 7,  8] | 262,144 B |       0 B | `Slot @ 0x00000` |
| Node 13 | Step [13, 14] |   5,120 B | 131,072 B | `Slot @ 0x20000` |
| Node 10 | Step [10, 11] | 131,072 B | 131,072 B | `Slot @ 0x20000` |
| Node  8 | Step [ 8,  9] | 262,144 B | 262,144 B | `Slot @ 0x40000` |

---

## 4. Systems Programming, Accounting & Safety Verification

1. **Accounting Disambiguation (Section 4.8):**
   - **Logical Intermediate Tensors:** Represents the total number of intermediate activation nodes created during model execution that require storage. Model A has 5 logical tensors, Model B has 9 logical tensors, and Model C has 7 logical tensors.
   - **Physical Reusable Arena Slots:** Represents the number of distinct spatial memory offset slots packed into the arena by interval coloring. Model A reuses 2 physical slots, Model B reuses 4 physical slots, and Model C reuses 3 physical slots.
   - **Alignment Overhead:** Every individual buffer is padded to a multiple of 64 bytes (`align_up(bytes)`), and every offset is 64-byte aligned (`offset % 64 == 0`).
   - **Excluded Memory:** Constant weights and biases (`CONST` nodes) are compiled into the native shared library `.rodata` section; caller input and final output tensors are allocated and owned by the caller.
   - **Calculation Type:** Peak activation memory is calculated analytically at compile-time via interval-coloring liveness analysis and verified directly via heap allocation tracking during execution.
2. **Heap Alignment Guarantee (D1):** Standard `std::vector` aligns the stack control block, not heap storage. ForgeTune v2.0 allocates intermediate arena buffers via `_aligned_malloc(alloc_bytes, 64)` (MSVC/Windows) or `posix_memalign` (POSIX), guaranteeing that base pointers and all 64-byte planned offsets are strictly 64-byte aligned to cache lines (`ptr % 64 == 0`).
3. **Thread-Safety & Re-Entrant Leasing (D2):** Storage is managed by `AlignedArenaPool` with thread-local pool instances and RAII `AlignedArenaPool::Lease`. Concurrent OS threads execute in total isolation without lock contention. Furthermore, nested or recursive same-thread calls safely lease independent arena blocks from the pool, eliminating same-thread re-entrancy memory corruption.
4. **Stack Overflow Mitigation:** On Windows environments with 1MB default thread stack limits, large convolutional feature maps at $B=128$ (exceeding 900 KB) would trigger `STATUS_STACK_OVERFLOW`. The heap-backed `AlignedArenaPool` completely eliminates stack allocation risks.
5. **Zero Heap Allocation Post-Warmup (D3):** Sizing the arena pool to $B_{max}=128$ capacity during initial warm-up ensures that all subsequent dynamic batch invocations ($B \in [1, 128]$) perform zero heap allocations, zero buffer resizing, and deterministic execution latency.
