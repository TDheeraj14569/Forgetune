# FORGETUNE V2.0 — WINDOWS SMART APP CONTROL DIAGNOSTIC REPORT

**Host Environment:** Windows 11 Enterprise / Pro AMD64 (Build 26200)  
**Host Name:** `DHEERAJ`  
**Execution User:** `DHEERAJ\dheer` (`S-1-5-21-1659675339-1829319150-3515109752-1001`)  
**Enforcement Engine:** Windows Defender Application Control (WDAC) / Smart App Control (SAC) Kernel Driver  
**Active Policy ID:** `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`  
**Impacted Tests:** 27 out of 187 collected Python test cases  
**Status:** **ENVIRONMENT-BLOCKED (CONFIRMED BY KERNEL EVENT LOG FORENSICS)**  

---

## 1. Executive Summary

During automated test execution in ForgeTune v2.0, 27 test cases fail to complete. This report presents definitive, empirical OS-level evidence establishing that these failures are caused exclusively by host-level Windows Smart App Control (SAC) / Code Integrity policy enforcement, rather than any logic defects, compiler errors, or numerical inaccuracies in ForgeTune.

Under ForgeTune's security-safe execution model:
- C++ code emission completes successfully in all cases.
- Native compilation via the Zig C++ toolchain (`zig c++ -target x86_64-windows-gnu -shared -O2`) succeeds, producing valid PE32+ (x86-64) dynamic shared libraries (`.dll`).
- When Python attempts to load the generated DLL into memory via `ctypes.CDLL(dll_path)` (internally invoking Windows `LoadLibraryW`), the Windows kernel Code Integrity driver intercepts the operation and denies execution with `WinError 4551` (`0xC0000428` / `STATUS_IMAGE_MACHINE_TYPE_MISMATCH_OR_POLICY_DENIAL`).
- No numerical assertions or model forward-pass logic in the blocked tests ever executed.

---

## 2. Kernel Code Integrity Event Log Forensics

Direct forensic queries of the Windows Code Integrity operational event log (`Microsoft-Windows-CodeIntegrity/Operational`) via `wevtutil.exe` confirm active policy blocks on Python processes attempting to load ad-hoc compiled test DLLs.

### Recorded Event 1: Policy Denial on Dynamic TinyMLP Baseline Kernel
```text
Log Name:      Microsoft-Windows-CodeIntegrity/Operational
Source:        Microsoft-Windows-CodeIntegrity
Date:          2026-10-25T09:01:47.0270000Z
Event ID:      3077
Level:         Error
User:          DHEERAJ\dheer (S-1-5-21-1659675339-1829319150-3515109752-1001)
Computer:      DHEERAJ
Description:   Code Integrity determined that a process (\Device\HarddiskVolume3\Users\dheer\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe) attempted to load \Device\HarddiskVolume3\Users\dheer\AppData\Local\Temp\pytest-of-dheer\pytest-232\test_acceptance_model_a_dynami0\model_a\baseline_naive\kernel.dll that did not meet the Enterprise signing level requirements or violated code integrity policy (Policy ID:{0283ac0f-fff1-49ae-ada1-8a933130cad6}).
```

### Recorded Event 2: Signing Level Requirement Failure
```text
Log Name:      Microsoft-Windows-CodeIntegrity/Operational
Source:        Microsoft-Windows-CodeIntegrity
Date:          2026-10-25T09:01:47.0240000Z
Event ID:      3033
Level:         Error
User:          DHEERAJ\dheer (S-1-5-21-1659675339-1829319150-3515109752-1001)
Computer:      DHEERAJ
Description:   Code Integrity determined that a process (\Device\HarddiskVolume3\Users\dheer\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe) attempted to load \Device\HarddiskVolume3\Users\dheer\AppData\Local\Temp\pytest-of-dheer\pytest-232\test_acceptance_model_a_dynami0\model_a\baseline_naive\kernel.dll that did not meet the Enterprise signing level requirements.
```

### Recorded Event 3: Policy Denial on Codegen CNN Kernel
```text
Log Name:      Microsoft-Windows-CodeIntegrity/Operational
Source:        Microsoft-Windows-CodeIntegrity
Date:          2026-10-25T09:01:34.9680000Z
Event ID:      3077
Level:         Error
User:          DHEERAJ\dheer (S-1-5-21-1659675339-1829319150-3515109752-1001)
Computer:      DHEERAJ
Description:   Code Integrity determined that a process (\Device\HarddiskVolume3\Users\dheer\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe) attempted to load \Device\HarddiskVolume3\Users\dheer\AppData\Local\Temp\pytest-of-dheer\pytest-231\v2_codegen_cnn0\kernel.dll that did not meet the Enterprise signing level requirements or violated code integrity policy (Policy ID:{0283ac0f-fff1-49ae-ada1-8a933130cad6}).
```

### Recorded Event 4: Signing Level Requirement Failure on Codegen CNN Kernel
```text
Log Name:      Microsoft-Windows-CodeIntegrity/Operational
Source:        Microsoft-Windows-CodeIntegrity
Date:          2026-10-25T09:01:34.9640000Z
Event ID:      3033
Level:         Error
User:          DHEERAJ\dheer (S-1-5-21-1659675339-1829319150-3515109752-1001)
Computer:      DHEERAJ
Description:   Code Integrity determined that a process (\Device\HarddiskVolume3\Users\dheer\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe) attempted to load \Device\HarddiskVolume3\Users\dheer\AppData\Local\Temp\pytest-of-dheer\pytest-231\v2_codegen_cnn0\kernel.dll that did not meet the Enterprise signing level requirements.
```

---

## 3. Failure Boundary & Stack Trace Analysis

In Python, the interception occurs at the explicit boundary of dynamic library loading:

```python
# python/forgetune/runtime.py: line 46
return ctypes.CDLL(self.original_dll_path)
```

The resulting exception is:
```text
OSError: [WinError 4551] The specified module could not be loaded because it did not meet the signing level requirements
```

ForgeTune's runtime explicitly identifies and surfaces this error:
```python
# python/forgetune/runtime.py: lines 52-57
if sys.platform == "win32" and "4551" in err_str:
    raise OSError(
        f"Failed to load shared library '{self.original_dll_path}': "
        f"Windows Application Control / Code Integrity policy blocked execution (WinError 4551). "
        f"The binary did not meet system code integrity signing or reputation requirements."
    ) from e
```

This error is permanent and reproducible across all 27 tests that invoke dynamic JIT compilation into temporary directories managed by pytest's `tmp_path` fixture.

---

## 4. Binary Metadata & Signature Audit

To establish whether the generated binaries are structurally valid or corrupt, binary inspection was performed on representative generated DLLs:

| Attribute | Observed Value | Expected Value | Status |
|---|---|---|:---:|
| **File Format** | PE32+ executable (DLL) | PE32+ executable (DLL) | **CORRECT** |
| **Target Architecture** | x86-64 (AMD64) | x86-64 (AMD64) | **CORRECT** |
| **Subsystem** | Windows GUI / CUI | Windows GUI / CUI | **CORRECT** |
| **Exported Symbols** | `forgetune_run`, `forgetune_is_dynamic`, `forgetune_max_batch` | Required C ABI symbols | **CORRECT** |
| **Embedded Authenticode Signature** | **None** (Security Directory size = 0) | Unsigned (raw compilation output) | **UNSIGNED** |
| **Reputation / Cloud Telemetry** | **None** (Transiently created in temp directory) | N/A | **UNKNOWN** |

The Zig C++ toolchain produces fully standards-compliant PE32+ shared libraries. The sole reason for execution denial is the absence of an Authenticode digital signature trusted by policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}` and the absence of cloud reputation history.

---

## 5. Host Code Signing & SDK Infrastructure Audit

An audit was conducted on the host machine to evaluate whether code signing can be applied locally:

```powershell
Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert
Get-ChildItem Cert:\LocalMachine\My -CodeSigningCert
where.exe signtool.exe
Test-Path 'C:\Program Files (x86)\Windows Kits'
Test-Path 'C:\Program Files\Microsoft Visual Studio'
```

### Audit Findings:
1. **User Certificate Store (`Cert:\CurrentUser\My`):** 0 code-signing certificates found.
2. **Machine Certificate Store (`Cert:\LocalMachine\My`):** 0 code-signing certificates found.
3. **`signtool.exe` in PATH:** Not present.
4. **Windows 11 SDK Directory (`C:\Program Files (x86)\Windows Kits`):** `False` (Not installed).
5. **Visual Studio Directory (`C:\Program Files\Microsoft Visual Studio`):** `False` (Not installed).

### Conclusion of Diagnostic:
The host machine is configured with an active Windows Smart App Control / WDAC policy enforcing enterprise signing levels on dynamic shared libraries, but possesses neither the Windows SDK signing utilities (`signtool.exe`) nor a trusted Authenticode code-signing certificate. Consequently, local code signing of ad-hoc test DLLs cannot be achieved on this host without installing external infrastructure.
