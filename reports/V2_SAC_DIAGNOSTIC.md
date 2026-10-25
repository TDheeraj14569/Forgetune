# FORGETUNE V2.0 — WINDOWS APPLICATION CONTROL / SAC DIAGNOSTIC REPORT

**Host Environment:** Windows 11 Enterprise / Pro AMD64 (Build 26200)  
**Host Name:** `DHEERAJ`  
**Execution User:** `DHEERAJ\dheer` (`S-1-5-21-1659675339-1829319150-3515109752-1001`)  
**Enforcement Subsystem:** Windows Defender Application Control (WDAC) / Smart App Control (SAC) Kernel Code Integrity Driver  
**Active Policy ID:** `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`  
**Impacted Objects:** Dynamically emitted unsigned test DLLs in newly created/transient directories  
**Audit Classification:** Technical Diagnostic Forensic Record  

---

## 1. Executive Summary

During automated testing and runtime code generation in ForgeTune v2.0, certain newly compiled dynamic shared libraries fail to load when invoked via Python's `ctypes.CDLL(dll_path)`. This report establishes the exact forensic diagnosis directly from Windows API errors and the Windows Code Integrity operational event logs.

### Key Forensic Findings:
1. **Compilation Phase:** The Zig C++ toolchain (`zig c++ -target x86_64-windows-gnu -shared -O2`) succeeds in 100% of cases, emitting valid, uncorrupted PE32+ (x86-64) dynamic shared libraries.
2. **Interception Stage:** The failure occurs during the operating system dynamic linking stage inside `LoadLibraryW()`, invoked by `ctypes.CDLL()`.
3. **Win32 Error Code:** `4551` (`ERROR_APP_CONTROL_BLOCKED`).
4. **Win32 System Error Message:** `"An Application Control policy has blocked this file."`
5. **Underlying NTSTATUS:** Neither the Python runtime exception nor the Windows Code Integrity operational event log provides an explicit underlying NTSTATUS code string. Prior references equating `WinError 4551` with machine-type mismatch (`0xC0000428` or `STATUS_IMAGE_MACHINE_TYPE_MISMATCH`) were incorrect conflations; `4551` is specifically `ERROR_APP_CONTROL_BLOCKED`.
6. **Kernel Log Evidence:** Event IDs `3077` and `3033` in `Microsoft-Windows-CodeIntegrity/Operational` explicitly record kernel policy enforcement against Policy ID `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`.
7. **Root Cause:** The binaries are unsigned (no Authenticode `WIN_CERTIFICATE` entry in the PE header) and have not established cloud reputation history within Windows Smart App Control / SmartScreen cloud telemetry.

---

## 2. Kernel Code Integrity Operational Event Log Evidence

Direct queries to `Microsoft-Windows-CodeIntegrity/Operational` via `wevtutil.exe` captured the following exact kernel events:

### Event ID 3077: Code Integrity Policy Block
```text
Log Name:      Microsoft-Windows-CodeIntegrity/Operational
Source:        Microsoft-Windows-CodeIntegrity
Date:          2026-10-25T10:17:14.7150000Z
Event ID:      3077
Level:         Error
User:          DHEERAJ\dheer (S-1-5-21-1659675339-1829319150-3515109752-1001)
Computer:      DHEERAJ
Description:   Code Integrity determined that a process (\Device\HarddiskVolume3\Users\dheer\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe) attempted to load \Device\HarddiskVolume3\Users\dheer\Desktop\Forgetune\build\artifacts\repro_2\baseline\kernel.dll that did not meet the Enterprise signing level requirements or violated code integrity policy (Policy ID:{0283ac0f-fff1-49ae-ada1-8a933130cad6}).
```

### Event ID 3033: Signing Level Requirement Failure
```text
Log Name:      Microsoft-Windows-CodeIntegrity/Operational
Source:        Microsoft-Windows-CodeIntegrity
Date:          2026-10-25T10:17:14.7110000Z
Event ID:      3033
Level:         Error
User:          DHEERAJ\dheer (S-1-5-21-1659675339-1829319150-3515109752-1001)
Computer:      DHEERAJ
Description:   Code Integrity determined that a process (\Device\HarddiskVolume3\Users\dheer\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe) attempted to load \Device\HarddiskVolume3\Users\dheer\Desktop\Forgetune\build\artifacts\repro_2\baseline\kernel.dll that did not meet the Enterprise signing level requirements.
```

---

## 3. Python Failure Boundary & Stack Trace

In Python, the interception occurs at the explicit boundary of dynamic library loading:

```python
# python/forgetune/runtime.py: line 46
return ctypes.CDLL(self.original_dll_path)
```

The resulting exception is:
```text
OSError: [WinError 4551] An Application Control policy has blocked this file
```

ForgeTune's runtime explicitly catches this system error and propagates an actionable diagnostic:
```python
# python/forgetune/runtime.py: lines 52-57
if sys.platform == "win32" and "4551" in err_str:
    raise OSError(
        f"Failed to load shared library '{self.original_dll_path}': "
        f"Windows Application Control / Code Integrity policy blocked execution (WinError 4551). "
        f"The binary did not meet system code integrity signing or reputation requirements."
    ) from e
```

---

## 4. Binary Metadata & Structure

Forensic inspection was conducted on the generated dynamic libraries:

| Attribute | Verified Value | Expected Specification | Audit Conclusion |
|---|---|---|:---:|
| **File Format** | PE32+ (64-bit) | PE32+ (64-bit) | **VALID** |
| **Machine Architecture** | `IMAGE_FILE_MACHINE_AMD64` (`0x8664`) | x86-64 | **VALID** |
| **Subsystem** | `IMAGE_SUBSYSTEM_WINDOWS_CUI` / `GUI` | Windows standard | **VALID** |
| **Export Directory** | Exports `forgetune_run`, `forgetune_is_dynamic`, `forgetune_max_batch` | Expected C ABI entry points | **VALID** |
| **Security Directory** | VirtualAddress = 0, Size = 0 | Unsigned binary | **UNSIGNED** |
| **Authenticode Signature** | **None** | Valid Authenticode certificate | **ABSENT** |
| **Cloud Reputation** | Novel hash | Established reputation | **NOVEL (UNTRUSTED)** |

---

## 5. Host Code-Signing Environment Audit

An audit of the host environment confirmed:
1. `Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert`: 0 certificates found.
2. `Get-ChildItem Cert:\LocalMachine\My -CodeSigningCert`: 0 certificates found.
3. `where.exe signtool.exe`: Not found in system PATH.
4. `C:\Program Files (x86)\Windows Kits`: False (Windows SDK not installed).
5. `C:\Program Files\Microsoft Visual Studio`: False (Visual Studio not installed).

### Conclusion:
The host operates under Windows Smart App Control / Code Integrity Policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`, but lacks the local PKI infrastructure and SDK tools required for Authenticode code signing.
