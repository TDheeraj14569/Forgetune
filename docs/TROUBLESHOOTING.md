# ForgeTune Troubleshooting Guide

## 1. Windows Application Control (WDAC / AppLocker) Error 4551

### Symptom:
`OSError: [WinError 4551] An Application Control policy has blocked this file`

### Underlying Cause:
Windows Defender Application Control or SmartScreen policy blocks executing or dynamically loading unsigned DLLs located in user directories (such as the Desktop).

### Solution in ForgeTune:
ForgeTune automatically stages compiled shared libraries to `%TEMP%\forgetune_runtime` and applies PowerShell `Unblock-File` stream unblocking before loading via `ctypes.CDLL`. This ensures seamless execution on hardened enterprise or locked Windows environments without requiring administrator privileges.

## 2. DLL File In-Use Lock on Windows

### Symptom:
`lld-link: error: failed to write output '...dll': Permission denied`

### Underlying Cause:
The Windows kernel places a mandatory lock on any `.dll` file mapped into a process by `LoadLibrary`.

### Solution in ForgeTune:
`NativeRuntime` copies the generated `.dll` to a uniquely named temporary file before loading it. The original build artifact on disk is never locked and can be recompiled or overwritten without conflict.
