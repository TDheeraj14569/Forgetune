# FORGETUNE V2.0 — WINDOWS SMART APP CONTROL REMEDIATION & CI EXECUTION PLAN

**Document Version:** 2.0.0-FINAL  
**Auditor:** Senior Windows Systems, CI/CD, & Compiler Verification Engineer  
**Date:** 2026-10-25  
**Target:** Policy-Compliant Resolution of SAC Interceptions on Novel Dynamic DLLs  

---

## 1. Executive Summary

In ForgeTune v2.0 on Windows 11 AMD64 (Build 26200), newly compiled dynamic shared libraries (`.dll`) in transient or novel directory paths are intercepted by the Windows kernel Code Integrity driver (`WinError 4551`: `"An Application Control policy has blocked this file."`, Policy ID: `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`).

This report documents the policy-compliant remediation pathways, details the implemented Windows CI workflow, presents local validation of the CI probe, and outlines the exact user actions required to trigger the remote verification run.

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

### 3.1 Technical Strategy
In standard software development workflows for JIT compilers, runtime code generators, and dynamic execution engines (e.g., PyTorch, LLVM, Julia, V8), test suites generating dynamic shared libraries in temporary directories are executed in environments configured for development/testing:
1. **Windows CI Runners (e.g., GitHub Actions `windows-latest`, Azure DevOps):** Standard virtual machines where Windows Defender Application Control / Smart App Control operates in Audit Mode, allowing dynamic binaries to load while logging integrity events without blocking execution.
2. **Dedicated Test VM:** An enterprise test environment where a supplemental WDAC policy is deployed granting execution rights to binaries located in designated development workspace paths.

### 3.2 Implemented Windows CI Workflow (`.github/workflows/windows_verification.yml`)
The workflow has been fully authored, verified, and placed in `.github/workflows/windows_verification.yml`. It implements:
1. Supported Windows runner: `runs-on: windows-latest`.
2. Explicit Python 3.12 and pinned dependency installation (`torch`, `numpy`, `pytest`, `pybind11`, `ziglang`).
3. Complete native compilation from source: builds `bin/forgetune_test_suite.exe`, `bin/forgetune_engine.exe`, and `python/forgetune/_forgetune_cpp.pyd`.
4. Execution of the dedicated dynamic loading probe: `python scripts/ci_probe.py` verifying direct `ctypes.CDLL` loading of unsigned JIT DLLs.
5. Execution of all 13 native C++ test functions directly via `bin/forgetune_test_suite.exe`.
6. Execution of all 187 Python test cases via `pytest -v --junitxml=reports/junit_report.xml tests/`.
7. Execution of models acceptance, dynamic-batch, and adversarial suites.
8. Execution of the Three-Round Master Verification Protocol (`scripts/v2_verify_all.py --mode=full`).
9. Execution of the Paired Performance Benchmark (`scripts/run_v2_paired_benchmark.py`).
10. Preservation of true exit codes (no `continue-on-error`, no swallowed exceptions).
11. Upload of full test logs, manifests, and benchmark outputs as GitHub Actions artifacts.

### 3.3 Local Validation of Dynamic Library Probe (`scripts/ci_probe.py`)
To verify that the compilation and ctypes loading pipeline works as intended on candidate runners, `scripts/ci_probe.py` was executed locally:
```powershell
.venv\Scripts\python.exe scripts/ci_probe.py
```
**Output:**
```text
Loading unsigned test DLL from: C:\Users\dheer\Desktop\Forgetune\build\ci_probe\probe.dll
[+] CI probe SUCCESS: C:\Users\dheer\Desktop\Forgetune\build\ci_probe\probe.dll loaded and executed cleanly (result=42)
```

---

## 4. User Action Required to Trigger Remote CI Execution

Per Rule 2:
> *If the workflow exists only in the local working tree and no remote run can be initiated without pushing, do not push automatically. Validate the workflow locally, produce the exact changes and instructions needed, and clearly identify the user action required to trigger the remote run.*

Because no remote Git repository is currently configured (`git remote -v` returned empty), the following explicit user action is required to trigger remote execution:

1. **Add the authorized GitHub remote repository:**
   ```powershell
   git remote add origin https://github.com/<ORGANIZATION>/Forgetune.git
   ```
2. **Push branch `v2.0-dev` to the remote:**
   ```powershell
   git push -u origin v2.0-dev
   ```
3. **Monitor CI execution:**
   * Navigate to `https://github.com/<ORGANIZATION>/Forgetune/actions`.
   * The `ForgeTune v2.0 Windows Verification CI` workflow will trigger automatically on push.
   * Verify all 12 steps complete with exit code 0.
   * Download the workflow artifact `forgetune-v2-windows-verification-artifacts` to inspect raw JUnit logs.
4. **Record the Workflow Run Evidence:**
   Upon completion, record:
   - Workflow Run URL: `https://github.com/<ORGANIZATION>/Forgetune/actions/runs/<RUN_ID>`
   - Tested Commit SHA: `7d2bf316773b92ccb5b18d0f05fae85e69851276` (or updated HEAD)
   - Runner OS Image: e.g., `Windows Server 2022 (Image 20241020.1.0)`
   - Artifact URI & Exit Code: `0` (Success).

---

## 5. Security Non-Evasion Policy

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

## 6. Current Defensible Resolution Status

1. **Local Host State:** All 187 Python test cases execute and pass assertions cleanly in the test suite. However, an independent clean-build reproducibility DLL in `build/artifacts/repro_2` was intercepted by host Smart App Control (`WinError 4551`), confirming active SAC policy enforcement on novel binaries and preventing the Round 3 bitwise reproducibility comparison from completing locally.
2. **Release Gate Status:**
   - **Gate 1 (Functional Correctness):** Remains on **HOLD** pending execution on the authorized Windows CI runner.
   - **Gate 2 (Legacy & ABI Compatibility):** Remains on **HOLD** pending execution on the authorized Windows CI runner.
   - **Gate 7 (Performance Parity & Non-Inferiority):** Remains on **HOLD** due to inconclusive two-sided TOST equivalence ($\pm 5\%$) across $N=100$ paired blocks ($90\%\ \text{CI} = [16.65, 30.89]\ \mu\text{s}$).
   - **Gate 12 (Overall Technical Readiness):** Remains on **HOLD** staged on `v2.0-dev`.
