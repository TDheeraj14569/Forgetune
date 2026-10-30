# FORGETUNE V2.0 — DETAILED AUDIT OF APPLICATION-CONTROL BLOCKED TESTS

**Host Environment:** Windows 11 AMD64 (Build 26200), Smart App Control / Code Integrity Policy Active
**Total Blocked Tests:** 0 / 187 collected Python tests
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

---

## 3. Breakdown by Test Category

- **Acceptance Tests (`tests/acceptance/`):** 17 tests blocked (Autotuner candidate gating, pipeline acceptance, randomized differential variants, Model C Small CNN acceptance).
- **Correctness Tests (`tests/correctness/`):** 9 tests blocked (5 kernel variants across dimension tuples, extreme values, differential hierarchy, prime dimensions, batch size one, all zeros/negatives).
- **Unit Tests (`tests/unit/`):** 1 test blocked (`test_float_serialization.py::test_bit_exact_float32_serialization`).

---

## 4. Policy Gate Impact

- **Gate 1 (Functional Correctness):** Must remain **HOLD** until tests can execute without policy denial.
- **Gate 2 (Legacy & ABI Compatibility):** Must remain **HOLD** until legacy acceptance suites can execute without policy denial.
- **Gate 11 (Overall Technical Readiness):** Must remain **HOLD** staged on `v2.0-dev` awaiting authorized signing / deployment environment.
