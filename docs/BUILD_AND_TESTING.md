# ForgeTune v2.0 Build & Testing Guide

This guide describes how to build, run tests, and execute verification suites for ForgeTune v2.0.

---

## 1. Environment & Prerequisites

- **Operating System:** Windows 10/11 (x86_64) or Linux (x86_64)
- **Python:** Python 3.11 or 3.12 (tested with Python 3.12.14)
- **PyTorch:** PyTorch >= 2.1 (tested with PyTorch 2.14.1+cpu)
- **Native Compiler:** Zig C++ toolchain (`ziglang` package, Zig 0.16.0 targeting `x86_64-windows-gnu`)
- **Key Python Packages:** `torch`, `numpy`, `pytest`, `pybind11`, `ziglang`

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

ForgeTune builds all native components directly using Zig C++:

### 3.1 Compile C++ Test Suite
```powershell
.venv\Lib\site-packages\ziglang\zig.exe c++ -target x86_64-windows-gnu -o bin/forgetune_test_suite.exe src/ir.cpp src/graph_verifier.cpp src/interpreter.cpp src/pass_manager.cpp src/optimizer.cpp src/fusion.cpp src/codegen.cpp src/json_io.cpp src/memory_planner.cpp tests/cpp/test_main.cpp -I include -std=c++17 -O2 -Wall -Wextra
```

### 3.2 Compile C++ Engine CLI
```powershell
.venv\Lib\site-packages\ziglang\zig.exe c++ -target x86_64-windows-gnu -o bin/forgetune_engine.exe src/ir.cpp src/graph_verifier.cpp src/interpreter.cpp src/pass_manager.cpp src/optimizer.cpp src/fusion.cpp src/codegen.cpp src/json_io.cpp src/memory_planner.cpp src/main.cpp -I include -std=c++17 -O2 -Wall -Wextra
```

### 3.3 Compile pybind11 Python Extension Module
```powershell
.venv\Lib\site-packages\ziglang\zig.exe c++ -target x86_64-windows-gnu -shared -o python/forgetune/_forgetune_cpp.pyd src/bindings.cpp src/ir.cpp src/graph_verifier.cpp src/interpreter.cpp src/pass_manager.cpp src/optimizer.cpp src/fusion.cpp src/codegen.cpp src/json_io.cpp src/memory_planner.cpp -I include -I <Python_Include_Path> -I <Pybind11_Include_Path> <Python_Lib_Path> -std=c++17 -O2 -Wall -Wextra
```

---

## 4. Running Tests

### 4.1 Native C++ Test Suite (13 Tests)
```powershell
bin\forgetune_test_suite.exe
```

### 4.2 Full Python Test Suite (229 Tests)
```powershell
$env:PYTHONPATH='python;.'
pytest -v tests/
```

### 4.3 Focused Acceptance Suites
```powershell
# Run v2 model acceptance tests
pytest -v tests/acceptance/test_v2_models_acceptance.py

# Run adversarial boundary and dual ABI tests
pytest -v tests/acceptance/test_v2_adversarial_boundary.py

# Run dynamic codegen runtime tests
pytest -v tests/unit/test_v2_codegen_runtime.py
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
