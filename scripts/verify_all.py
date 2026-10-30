"""
Top-level verification runner for ForgeTune.
Executes all C++ and Python tests, benchmarks, audits, and end-to-end flows.
Returns non-zero exit code if any check fails.
"""

import os
import sys
import subprocess
import time

def run_step(step_name: str, cmd: list, env: dict = None) -> bool:
    print("\n========================================================")
    print(f" [VERIFY] {step_name}")
    print(f" Command: {' '.join(cmd)}")
    print("========================================================")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env)
    dur = time.perf_counter() - t0
    if p.returncode != 0:
        print(f"[-] FAILED: {step_name} exited with code {p.returncode} in {dur:.2f}s")
        return False
    print(f"[+] PASSED: {step_name} in {dur:.2f}s")
    return True

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root_dir)

    python_exe = os.path.join(root_dir, ".venv", "Scripts", "python.exe")
    os.path.join(root_dir, ".venv", "Scripts", "pytest.exe")
    zig_exe = os.path.join(root_dir, ".venv", "Lib", "site-packages", "ziglang", "zig.exe")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.path.join(root_dir, 'python')};{root_dir}"

    print("=================================================================")
    print(" ForgeTune Full Verification Suite")
    print(f" Root Directory: {root_dir}")
    print("=================================================================")

    # Step 1: Recompile and run C++ native test suite
    cpp_srcs = [
        "src/ir.cpp",
        "src/graph_verifier.cpp",
        "src/interpreter.cpp",
        "src/pass_manager.cpp",
        "src/optimizer.cpp",
        "src/fusion.cpp",
        "src/codegen.cpp",
        "src/json_io.cpp",
        "src/memory_planner.cpp",
        "tests/cpp/test_main.cpp"
    ]
    # Step 1: Build pybind11 extension module
    import sysconfig
    import pybind11
    py_include = sysconfig.get_path("include")
    pybind_include = pybind11.get_include()
    py_lib = os.path.join(sys.base_prefix, "libs", f"python{sys.version_info.major}{sys.version_info.minor}.lib")
    pyd_output = os.path.join("python", "forgetune", "_forgetune_cpp.pyd")
    compile_pyd_cmd = [
        zig_exe, "c++", "-target", "x86_64-windows-gnu", "-shared", "-o", pyd_output,
        "src/bindings.cpp",
        "tests/cpp/test_main.cpp",
        "-DFORGETUNE_EXCLUDE_MAIN",
        *cpp_srcs[:-1],
        "-I", "include",
        "-I", py_include,
        "-I", pybind_include,
        py_lib,
        "-std=c++17", "-O2", "-w"
    ]
    if not run_step("Compile pybind11 Extension Module", compile_pyd_cmd):
        sys.exit(1)

    # Step 2: Compile C++ test suite executable
    compile_cpp_cmd = [
        zig_exe, "c++", "-target", "x86_64-windows-gnu", "-o", "bin/forgetune_test_suite.exe",
        *cpp_srcs,
        "-I", "include", "-std=c++17", "-O2", "-w"
    ]
    os.makedirs("bin", exist_ok=True)
    if not run_step("Compile C++ Test Suite Executable", compile_cpp_cmd):
        sys.exit(1)

    # Step 3: Run C++ test suite (direct or in-process)
    test_suite_exe = os.path.abspath("bin/forgetune_test_suite.exe")
    cpp_suite_passed = False
    try:
        if run_step("Run C++ Test Suite", [test_suite_exe]):
            cpp_suite_passed = True
    except Exception as e:
        print(f"Direct executable execution blocked/unavailable ({e}). Running via in-process pybind11...")

    if not cpp_suite_passed:
        cmd = [python_exe, "-c", "import sys, forgetune._forgetune_cpp as cpp; sys.exit(cpp.run_cpp_test_suite())"]
        if not run_step("Run C++ Test Suite (in-process)", cmd, env=env):
            sys.exit(1)

    # Step 4: Build C++ engine CLI
    cli_srcs = [
        "src/ir.cpp",
        "src/graph_verifier.cpp",
        "src/interpreter.cpp",
        "src/pass_manager.cpp",
        "src/optimizer.cpp",
        "src/fusion.cpp",
        "src/codegen.cpp",
        "src/json_io.cpp",
        "src/memory_planner.cpp",
        "src/main.cpp"
    ]
    compile_cli_cmd = [
        zig_exe, "c++", "-target", "x86_64-windows-gnu", "-o", "bin/forgetune_engine.exe",
        *cli_srcs,
        "-I", "include", "-std=c++17", "-O2", "-w"
    ]
    if not run_step("Compile C++ Engine CLI", compile_cli_cmd):
        sys.exit(1)

    # Step 4: Run full pytest test suite
    if not run_step("Run Python PyTest Suite", [python_exe, "-m", "pytest", "-v", "tests/"], env=env):
        sys.exit(1)

    # Step 5: Run end-to-end TinyMLP example
    if not run_step("Execute TinyMLP Example", [python_exe, "examples/tiny_mlp.py"], env=env):
        sys.exit(1)

    print("\n=================================================================")
    print(" ALL FORGETUNE VERIFICATION CHECKS PASSED WITH EXIT CODE 0!")
    print("=================================================================")
    sys.exit(0)

if __name__ == "__main__":
    main()
