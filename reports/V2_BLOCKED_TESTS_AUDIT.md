# FORGETUNE V2.0 — DETAILED AUDIT OF APPLICATION-CONTROL BLOCKED TESTS

**Host Environment:** Windows 11 AMD64 (Build 26200), Smart App Control / Code Integrity Policy Active
**Total Blocked Tests:** 27 / 187 collected Python tests
**Policy Error Code:** `WinError 4551` (`STATUS_IMAGE_MACHINE_TYPE_MISMATCH_OR_POLICY_DENIAL` / Code Integrity Policy Block)

---

## 1. Executive Summary & Root Cause Analysis

Under direct, security-safe library loading (`python/forgetune/runtime.py`) without policy evasion workarounds (no `Unblock-File`, no NTFS `:Zone.Identifier` stream stripping, no randomized filename rotation):
1. **Emitted C++ Source Emission:** Successfully executed (Stage 3).
2. **Zig Native Toolchain Compilation:** Successfully executed via `zig c++ -shared -O2` producing valid PE/COFF dynamic library files (Stage 6).
3. **Interception Point:** OS dynamic linking (`ctypes.CDLL(dll_path)` -> Windows `LoadLibraryW`). Because the generated `.dll` in `tmp_path` is unsigned and lacks reputation history, the Windows kernel Code Integrity / Smart App Control driver denies execution with `WinError 4551`.
4. **Assertion Execution Status:** **None of the intended numerical assertions or model outputs executed.** Testing was intercepted strictly at the shared library loading boundary before kernel invocation.
5. **Required Environment for Resolution:**
   - A development/CI environment with digital code-signing infrastructure (`signtool.exe` with a trusted code-signing certificate), OR
   - An authorized test environment where Windows App Control (SAC / WDAC) is configured in Audit mode or explicitly allows binaries in configured temporary workspace paths.

---

## 2. Granular Inventory of All 27 Blocked Tests

| Index | Test Identifier | Intercepted Artifact | Build Succeeded | Assertions Executed | Exact Denial Diagnosis |
|:---:|:---|:---|:---:|:---:|:---|
| **01** | `tests/acceptance/test_autotuner_integrity.py::test_acceptance_correctness_gating_rejects_failing_candidate` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **02** | `tests/acceptance/test_pipeline_e2e_acceptance.py::test_acceptance_single_linear_with_bias` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **03** | `tests/acceptance/test_pipeline_e2e_acceptance.py::test_acceptance_three_layer_mlp_with_relu_and_fusion` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **04** | `tests/acceptance/test_randomized_differential.py::test_acceptance_randomized_matmul_all_unfused_variants[naive_ijk-3-7-17-prime_small]` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **05** | `tests/acceptance/test_randomized_differential.py::test_acceptance_randomized_matmul_all_unfused_variants[naive_ijk-7-8-15-subtile_15]` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **06** | `tests/acceptance/test_randomized_differential.py::test_acceptance_randomized_matmul_all_unfused_variants[naive_ijk-33-15-17-post_boundary_33]` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **07** | `tests/acceptance/test_randomized_differential.py::test_acceptance_randomized_matmul_all_unfused_variants[loop_ikj-1-1-1-scalar_1x1]` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **08** | `tests/acceptance/test_randomized_differential.py::test_acceptance_randomized_matmul_all_unfused_variants[loop_ikj-3-7-17-prime_small]` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **09** | `tests/acceptance/test_randomized_differential.py::test_acceptance_randomized_matmul_all_unfused_variants[loop_ikj-32-32-32-exact_tile_32]` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **10** | `tests/acceptance/test_randomized_differential.py::test_acceptance_randomized_matmul_all_unfused_variants[loop_ikj-33-15-17-post_boundary_33]` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **11** | `tests/acceptance/test_randomized_differential.py::test_acceptance_randomized_matmul_all_unfused_variants[loop_ikj-64-8-4-rectangular_tall]` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **12** | `tests/acceptance/test_randomized_differential.py::test_acceptance_randomized_matmul_all_unfused_variants[tiled-33-15-17-post_boundary_33]` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **13** | `tests/acceptance/test_randomized_differential.py::test_acceptance_randomized_matmul_all_unfused_variants[tiled-64-8-4-rectangular_tall]` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **14** | `tests/acceptance/test_randomized_differential.py::test_acceptance_randomized_fused_variants[fused_ikj-16-16-16-tile16_fused]` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **15** | `tests/acceptance/test_randomized_differential.py::test_acceptance_randomized_fused_variants[fused_tiled-16-16-16-tile16_fused]` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **16** | `tests/acceptance/test_randomized_differential.py::test_acceptance_extreme_values_stress` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **17** | `tests/acceptance/test_v2_models_acceptance.py::test_acceptance_model_c_small_cnn` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **18** | `tests/correctness/test_all_variants.py::test_all_five_kernel_variants[17-31-23]` | `tiled_17_31_23.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **19** | `tests/correctness/test_all_variants.py::test_all_five_kernel_variants[16-16-16]` | `fused_ikj_16_16_16.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **20** | `tests/correctness/test_all_variants.py::test_all_five_kernel_variants[3-5-7]` | `tiled_3_5_7.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **21** | `tests/correctness/test_all_variants.py::test_all_five_kernel_variants[1-48-24]` | `naive_1_48_24.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **22** | `tests/correctness/test_all_variants.py::test_extreme_inputs_all_variants` | `extreme_fused.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **23** | `tests/correctness/test_differential.py::test_full_differential_hierarchy` | `tiled_16.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **24** | `tests/correctness/test_edge_cases.py::test_prime_dimensions_and_partial_tiles` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **25** | `tests/correctness/test_edge_cases.py::test_batch_size_one` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **26** | `tests/correctness/test_edge_cases.py::test_all_zeros_and_all_negatives` | `kernel.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |
| **27** | `tests/unit/test_float_serialization.py::test_bit_exact_float32_serialization` | `float_test.dll` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |

---

## 3. Breakdown by Test Category

- **Acceptance Tests (`tests/acceptance/`):** 17 tests blocked (Autotuner candidate gating, pipeline acceptance, randomized differential variants, Model C Small CNN acceptance).
- **Correctness Tests (`tests/correctness/`):** 9 tests blocked (5 kernel variants across dimension tuples, extreme values, differential hierarchy, prime dimensions, batch size one, all zeros/negatives).
- **Unit Tests (`tests/unit/`):** 1 test blocked (`test_float_serialization.py::test_bit_exact_float32_serialization`).

---

## 4. Policy Gate Impact (Historical Evaluation)

- **Gate 1 (Functional Correctness):** Held on **HOLD** historically until tests could execute without policy denial.
- **Gate 2 (Legacy & ABI Compatibility):** Held on **HOLD** historically until legacy acceptance suites could execute without policy denial.
- **Gate 11 (Overall Technical Readiness):** Held on **HOLD** historically awaiting authorized deployment environment.

---

## 5. Historical Context & Resolution Status (Superseded)

> [!NOTE]
> This document preserves the historical audit of the 27 tests intercepted under Windows Smart App Control (`WinError 4551`) prior to CI deployment and host policy adjustment.
>
> **Current Verified Resolution:**
> 1. **Remote Windows CI:** In authorized GitHub Actions Windows CI ([Run 40174972712](https://github.com/TDheeraj14569/Forgetune/actions/runs/40174972712)), all 187 collected Python tests and all 13 native C++ tests passed cleanly with 0 failures and 0 blocks.
> 2. **Local Workstation:** Following SAC policy deactivation by the workstation administrator, all 187 Python tests execute assertions to completion and pass.
> 3. **Gate Status:** Gates 1, 2, and 11 have been formally promoted to **PASS** in [reports/V2_FINAL_CLOSURE_AUDIT.md](file:///c:/Users/dheer/Desktop/Forgetune/reports/V2_FINAL_CLOSURE_AUDIT.md) and [reports/V2_REQUIREMENTS_MATRIX.md](file:///c:/Users/dheer/Desktop/Forgetune/reports/V2_REQUIREMENTS_MATRIX.md).
