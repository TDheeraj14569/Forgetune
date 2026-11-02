# ForgeTune v2.0 Build & Testing Guide

This guide describes how to build, run tests, and execute verification suites for ForgeTune v2.0.

---

## 1. Environment & Prerequisites

- **Operating System:** Windows 10/11 (x86_64) or Linux (Ubuntu 22.04+, x86_64)
- **Python Support & CI Verification Scope:**
  - **Python 3.12:** Full test suite (306 pytest items + 13 native C++ tests) and three-round master verification protocol in the primary CI workflow (tested with Python 3.12.14).
  - **Python 3.11:** Core acceptance suite and native C++ compatibility smoke verification in CI (tested with Python 3.11).
- **PyTorch:** PyTorch >= 2.1 (tested with PyTorch 2.14.1+cpu)
- **Native Toolchain:**
  - Zig C++ toolchain (`ziglang` package, Zig 0.16.0 targeting `x86_64-windows-gnu` or `x86_64-linux-gnu`)
  - Pure CMake 3.20+ with host compiler (Clang/GCC/MSVC) and CTest (verified in Linux CI)
- **Key Python Packages:** `torch`, `numpy`, `pytest`, `pybind11`, `ziglang`, `scipy`

---

## 2. Setting Up the Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install torch numpy pytest pybind11 ziglang
```

---

## 3. Building Compiler Binaries and Extensions

ForgeTune builds all native components cleanly using the cross-platform native build script:

```bash
# Compile native extension (_forgetune_cpp), engine CLI, and test suite with -Wall -Wextra
python scripts/build_native.py
```
This automatically compiles:
- `python/forgetune/_forgetune_cpp.pyd` (Windows) / `_forgetune_cpp.so` (Linux)
- `bin/forgetune_engine` and `bin/forgetune_engine.exe`
- `bin/forgetune_test_suite` and `bin/forgetune_test_suite.exe`

---

## 4. Running Tests

### 4.1 Native C++ Test Suite (13 Tests)
```bash
./bin/forgetune_test_suite      # Linux
# or .\bin\forgetune_test_suite.exe on Windows
```

### 4.2 Full Python Test Suite (306 Tests)
```bash
python -m pytest -v tests/
```

### 4.3 Focused Acceptance Suites
```bash
# Run v2 model acceptance tests
python -m pytest -v tests/acceptance/test_v2_models_acceptance.py

# Run adversarial boundary and dual ABI tests
python -m pytest -v tests/acceptance/test_v2_adversarial_boundary.py

# Run dynamic codegen runtime tests
python -m pytest -v tests/unit/test_v2_codegen_runtime.py
```

---

## 5. Master Verification Rounds

To execute the three complete verification rounds and produce all audit reports:

```powershell
$env:PYTHONPATH='python;.'
python scripts/v2_verify_all.py
```
This script runs:
- **Round 1:** Clean build from scratch, runs C++ suite, runs full PyTest suite, executes Models A, B, and C.
- **Round 2:** Runs adversarial boundary suite, acceptance models suite, benchmarks, and memory report.
- **Round 3:** Fresh clean rebuild and dual-compilation bitwise reproducibility check.
- Generates `reports/V2_FINAL_VERIFICATION.md`.
