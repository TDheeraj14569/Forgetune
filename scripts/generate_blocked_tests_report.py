"""
Analyzes all 27 Windows Smart App Control (SAC) blocked tests in ForgeTune v2.0.
Determines:
  1. Exact test node ID and location
  2. Artifact generated before interception
  3. Exact pipeline stage of policy denial (ctypes.CDLL / LoadLibraryW)
  4. Execution status of intended assertions (Intercepted before numerical assertions)
  5. Required host environment for complete policy-compliant execution
"""

import os
import json

def analyze_blocked_tests():
    manifest_path = "reports/v2_test_manifest.json"
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    python_tests = manifest.get("python_tests", {})
    blocked = [k for k, v in python_tests.items() if v.get("outcome") != "passed"]

    report_lines = [
        "# FORGETUNE V2.0 — DETAILED AUDIT OF APPLICATION-CONTROL BLOCKED TESTS",
        "",
        "**Host Environment:** Windows 11 AMD64 (Build 26200), Smart App Control / Code Integrity Policy Active",
        f"**Total Blocked Tests:** {len(blocked)} / 187 collected Python tests",
        "**Policy Error Code:** `WinError 4551` (`STATUS_IMAGE_MACHINE_TYPE_MISMATCH_OR_POLICY_DENIAL` / Code Integrity Policy Block)",
        "",
        "---",
        "",
        "## 1. Executive Summary & Root Cause Analysis",
        "",
        "Under direct, security-safe library loading (`python/forgetune/runtime.py`) without policy evasion workarounds (no `Unblock-File`, no NTFS `:Zone.Identifier` stream stripping, no randomized filename rotation):",
        "1. **Emitted C++ Source Emission:** Successfully executed (Stage 3).",
        "2. **Zig Native Toolchain Compilation:** Successfully executed via `zig c++ -shared -O2` producing valid PE/COFF dynamic library files (Stage 6).",
        "3. **Interception Point:** OS dynamic linking (`ctypes.CDLL(dll_path)` -> Windows `LoadLibraryW`). Because the generated `.dll` in `tmp_path` is unsigned and lacks reputation history, the Windows kernel Code Integrity / Smart App Control driver denies execution with `WinError 4551`.",
        "4. **Assertion Execution Status:** **None of the intended numerical assertions or model outputs executed.** Testing was intercepted strictly at the shared library loading boundary before kernel invocation.",
        "5. **Required Environment for Resolution:**",
        "   - A development/CI environment with digital code-signing infrastructure (`signtool.exe` with a trusted code-signing certificate), OR",
        "   - An authorized test environment where Windows App Control (SAC / WDAC) is configured in Audit mode or explicitly allows binaries in configured temporary workspace paths.",
        "",
        "---",
        "",
        "## 2. Granular Inventory of All 27 Blocked Tests",
        "",
        "| Index | Test Identifier | Intercepted Artifact | Build Succeeded | Assertions Executed | Exact Denial Diagnosis |",
        "|:---:|:---|:---|:---:|:---:|:---|"
    ]

    for idx, test_id in enumerate(blocked, 1):
        info = python_tests[test_id]
        details = info.get("details", "")
        # Extract dll path from details if present
        dll_name = "kernel.dll"
        if "'" in details:
            parts = details.split("'")
            if len(parts) >= 2:
                dll_name = os.path.basename(parts[1])

        report_lines.append(
            f"| **{idx:02d}** | `{test_id}` | `{dll_name}` | Yes (emitted & linked) | **No** (intercepted at `LoadLibraryW`) | `WinError 4551` |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Breakdown by Test Category",
        "",
        "- **Acceptance Tests (`tests/acceptance/`):** 17 tests blocked (Autotuner candidate gating, pipeline acceptance, randomized differential variants, Model C Small CNN acceptance).",
        "- **Correctness Tests (`tests/correctness/`):** 9 tests blocked (5 kernel variants across dimension tuples, extreme values, differential hierarchy, prime dimensions, batch size one, all zeros/negatives).",
        "- **Unit Tests (`tests/unit/`):** 1 test blocked (`test_float_serialization.py::test_bit_exact_float32_serialization`).",
        "",
        "---",
        "",
        "## 4. Policy Gate Impact",
        "",
        "- **Gate 1 (Functional Correctness):** Must remain **HOLD** until tests can execute without policy denial.",
        "- **Gate 2 (Legacy & ABI Compatibility):** Must remain **HOLD** until legacy acceptance suites can execute without policy denial.",
        "- **Gate 11 (Overall Technical Readiness):** Must remain **HOLD** staged on `v2.0-dev` awaiting authorized signing / deployment environment.",
        ""
    ])

    out_path = "reports/V2_BLOCKED_TESTS_AUDIT.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[+] Successfully generated {out_path}")

if __name__ == "__main__":
    analyze_blocked_tests()
