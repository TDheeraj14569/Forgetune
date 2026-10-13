# ForgeTune v2.0 Troubleshooting Guide

This guide covers common issues, root causes, and diagnostic steps when building, running, and benchmarking models in ForgeTune v2.0.

---

## 1. Windows Application Control (WDAC / AppLocker) Error 4551

### Symptom:
`OSError: [WinError 4551] An Application Control policy has blocked this file`

### Underlying Cause:
Windows Defender Application Control or SmartScreen policy evaluates newly created DLLs asynchronously and blocks loading unsigned binaries located in user directories.

### Built-in Solution:
`NativeRuntime` automatically copies compiled `.dll` libraries to `%TEMP%\forgetune_runtime`, strips the NTFS `Zone.Identifier` stream, and implements bounded retry logic with PowerShell `Unblock-File`.

---

## 2. Windows Thread Stack Overflow (`STATUS_STACK_OVERFLOW` 0xC00000FD)

### Symptom:
Process crashes silently with exit code `3221225725` (`0xC00000FD`) when running convolutional models at large batch sizes ($B=128$).

### Underlying Cause:
Windows allocates a default thread stack of only 1 MB. Allocating large intermediate activation tensors on the stack (e.g. Small CNN feature maps at $B=128$ exceeding 900 KB) overflows the thread stack.

### Built-in Solution:
ForgeTune v2.0 allocates all intermediate activation tensors inside a single heap-allocated `alignas(64) static thread_local std::vector<float> arena_storage`. It completely eliminates stack allocation risks while ensuring thread-safety and re-entrancy.

---

## 3. Dynamic Batch Size Rejection (`InvalidShapeError`)

### Symptom:
`forgetune.errors.InvalidShapeError: Dynamic batch size 129 out of valid range [1, 128]`

### Underlying Cause:
ForgeTune v2.0 contracts restrict dynamic batch sizes to the configured range $B \in [1, B_{max}]$ (default $B_{max} = 128$) to guarantee memory arena safety.

### Solution:
Ensure input tensor batch dimensions adhere to $1 \le B \le 128$. For applications requiring larger batch sizes, specify `max_batch_size` during compilation:
```python
compiled = compile_model(model, example_inputs, dynamic_batch=True, max_batch_size=256)
```

---

## 4. `ModuleNotFoundError: No module named 'forgetune'`

### Symptom:
`ModuleNotFoundError: No module named 'forgetune'` when running scripts or examples directly from the command line.

### Underlying Cause:
Python does not find the `python/` package directory in its default `sys.path`.

### Solution:
Set `PYTHONPATH` before executing:
```powershell
# PowerShell
$env:PYTHONPATH='python;.'
python examples/v2_dynamic_tinymlp.py
```
```bash
# Bash
PYTHONPATH=python:. python examples/v2_dynamic_tinymlp.py
```

---

## 5. Zig Toolchain Linking Errors on Windows

### Symptom:
`lld-link: error: undefined symbol: __chkstk` or target ABI incompatibilities.

### Solution:
Always pass `-target x86_64-windows-gnu` when using Zig C++ on Windows:
```powershell
.venv\Lib\site-packages\ziglang\zig.exe c++ -target x86_64-windows-gnu -o <binary> ...
```
