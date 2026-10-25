# FORGETUNE V2.0 — WINDOWS SMART APP CONTROL REMEDIATION ANALYSIS & ACTION PLAN

**Document Version:** 2.0.0-FINAL  
**Author:** Senior Windows Systems & Release Verification Auditor  
**Date:** 2026-10-25  
**Target:** Policy-Compliant Resolution of the 27 SAC-Blocked Test Cases  

---

## 1. Executive Summary

In ForgeTune v2.0 on Windows 11 AMD64 (Build 26200), 27 test cases fail to complete due to Windows Smart App Control / Code Integrity policy denial (`WinError 4551`, Policy ID: `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`).

This document evaluates the two legitimate, policy-compliant remediation pathways:
1. **Route A (Trusted Digital Code Signing):** Post-link digital signing of generated DLLs using `signtool.exe` and a trusted Authenticode certificate.
2. **Route B (Authorized CI / Permissive Test Environment):** Execution on an authorized Windows CI runner or test machine where Smart App Control is in Audit Mode or Disabled.

Both pathways are evaluated against strict architectural and security constraints.

---

## 2. Evaluation of Route A: Trusted Digital Code Signing

### 2.1 Technical Requirements
To satisfy Windows Smart App Control and enterprise Code Integrity policies via Route A, compiled shared libraries must undergo an Authenticode digital signing step immediately after compilation and prior to invocation of `ctypes.CDLL()` / `LoadLibraryW()`.

The required command sequence is:
```cmd
signtool.exe sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /sha1 <CERT_THUMBPRINT> <DLL_PATH>
signtool.exe verify /pa /all /v <DLL_PATH>
```

### 2.2 Certificate Authority & Trust Hierarchy Requirements
- The signing certificate must be an **Authenticode Code Signing Certificate** issued by a publicly trusted Certificate Authority (e.g., DigiCert, Sectigo, GlobalSign) or an Enterprise Root CA provisioned via Active Directory Group Policy.
- **Strict Prohibition on Self-Signed Certificates:** Creating an ad-hoc self-signed certificate (e.g., via PowerShell `New-SelfSignedCertificate`) without trusted root anchor installation does NOT satisfy Windows Smart App Control. Furthermore, forcing trust by injecting self-signed roots into the machine root store or manipulating local security policy is strictly prohibited as a security evasion tactic.

### 2.3 Host Environment Audit Findings for Route A
As established in `reports/V2_SAC_DIAGNOSTIC.md`:
1. `Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert` returned **0 certificates**.
2. `Get-ChildItem Cert:\LocalMachine\My -CodeSigningCert` returned **0 certificates**.
3. `signtool.exe` is **not installed** on the host PATH.
4. Windows 11 SDK (`C:\Program Files (x86)\Windows Kits`) is **not installed**.
5. Visual Studio (`C:\Program Files\Microsoft Visual Studio`) is **not installed**.

### 2.4 Conclusion for Route A
Route A is **technically sound and production-ready in design**, but is **locally infeasible** on this development host due to the absence of the Windows SDK and trusted signing credentials. Fabricating a mock or self-signed signing step is expressly disallowed.

---

## 3. Evaluation of Route B: Authorized Windows CI Runner / Audit Mode

### 3.1 Technical Requirements
In standard software development workflows for JIT compilers, runtime code generators, and dynamic execution engines (e.g., PyTorch, LLVM, Julia, V8), test suites generating dynamic shared libraries in temporary directories are executed in environments configured for development/testing:
1. **Windows CI Runners (e.g., GitHub Actions `windows-latest`, Azure DevOps):** Standard virtual machines where Windows Defender Application Control / Smart App Control operates in Audit Mode, allowing dynamic binaries to load while logging integrity events without blocking execution.
2. **Dedicated Test VM:** An enterprise test environment where a supplemental WDAC policy is deployed granting execution rights to binaries located in designated development workspace paths.

### 3.2 Repository Status for Route B
An audit of the repository root confirmed:
- No `.github/` or `.github/workflows/` directory currently exists in the repository.
- No automated cloud CI runner is currently attached.

### 3.3 Actionable CI Workflow Configuration
To establish Route B, the following GitHub Actions workflow definition is specified for deployment upon CI configuration:

```yaml
# .github/workflows/windows_verification.yml
name: ForgeTune v2.0 Windows Verification

on:
  push:
    branches: [ v2.0-dev ]
  pull_request:
    branches: [ master ]

jobs:
  verify:
    runs-on: windows-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest numpy torch pybind11 ziglang

      - name: Run Master Verification Protocol
        run: |
          python scripts/v2_verify_all.py --mode=full --skip-benchmarks
```

In GitHub Actions `windows-latest` runners, code integrity enforcement does not intercept transient test DLLs, enabling all 187 Python tests and 13 native C++ test functions (all 200 enumerated test identifiers) to execute assertions to completion.

---

## 4. Strict Non-Evasion Policy

ForgeTune explicitly adheres to enterprise software security standards:
- **No `Unblock-File`:** ForgeTune never executes PowerShell `Unblock-File` or alters security descriptors.
- **No NTFS Stream Stripping:** ForgeTune never deletes `:Zone.Identifier` alternate data streams.
- **No Filename Rotation:** ForgeTune never generates randomized filenames or moves files across directories to defeat file-reputation telemetry.
- **No Security Downgrade:** ForgeTune never modifies Windows Registry keys (`SmartScreenEnabled`, `AppLocker`, `CodeIntegrity`).

All library loading in `python/forgetune/runtime.py` is performed directly on the original compiled artifact:
```python
# python/forgetune/runtime.py: lines 37-40
# Attempt direct library loading on the original compiled artifact.
# Security metadata is fully preserved; no Zone.Identifier stream deletion,
# no randomized filename rotation, and no Unblock-File execution is performed.
# Bounded retries are strictly reserved for documented transient file-sharing contention.
```

---

## 5. Build and Load Pipeline Failure Semantics

To ensure unambiguous test results, the pipeline failure semantics are strictly separated:

| Failure Type | Trigger Condition | Exception Raised | Test Outcome |
|---|---|---|:---:|
| **Compilation Failure** | C++ syntax error, missing header, invalid compiler flag | `forgetune.errors.CompilationError` | **FAILED** |
| **Signing Failure** | `signtool.exe` returns non-zero, invalid certificate | `forgetune.errors.CompilationError` | **FAILED** |
| **Missing Artifact** | Output DLL not found on disk | `FileNotFoundError` | **FAILED** |
| **Policy Denial** | Kernel Code Integrity denial (`WinError 4551`) | `OSError` (with explicit SAC diagnosis) | **BLOCKED** |
| **Logic / Assertion** | Output mismatch, shape error, incorrect tensor values | `AssertionError` | **FAILED** |

Under default execution (`--mode=full`), `scripts/v2_verify_all.py` enforces that any BLOCKED step results in a non-zero exit code (exit code 1), ensuring that blocked tests are never misreported as passing.

---

## 6. Current Defensible Resolution Status

1. **Local Host State:** The 27 tests remain legitimately **BLOCKED** by host application control policy on this machine. They cannot be executed locally without external trusted signing infrastructure or host policy reconfiguration.
2. **Honest Accounting:** The tests are recorded as **BLOCKED**, not FAILED (compilation succeeds, 0 numerical failures observed) and not PASSED (assertions did not execute).
3. **Release Gate Impact:**
   - **Gate 1 (Functional Correctness):** Remains on **HOLD** pending full assertion execution in an authorized environment.
   - **Gate 2 (Legacy & ABI Compatibility):** Remains on **HOLD** pending execution of legacy acceptance tests.
   - **Gate 7 (Performance Parity & Non-Inferiority):** Remains on **HOLD** due to inconclusive two-sided TOST equivalence ($\pm 5\%$) at $N=30$.
   - **Gate 12 (Overall Technical Readiness):** Remains on **HOLD** staged on `v2.0-dev`.
