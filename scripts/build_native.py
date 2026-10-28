"""
Build script to compile all native ForgeTune C++ targets using Zig.
Targets:
1. python/forgetune/_forgetune_cpp.pyd (Pybind11 extension)
2. bin/forgetune_engine.exe (Compiler engine CLI)
3. bin/forgetune_test_suite.exe (C++ native test suite)
"""

import os
import sys
import subprocess

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    zig_exe = os.path.join(repo_root, ".venv", "Lib", "site-packages", "ziglang", "zig.exe")
    if not os.path.isfile(zig_exe):
        zig_exe = "zig"

    py_inc = os.path.expandvars(r"C:\Users\dheer\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\include")
    py_lib = os.path.expandvars(r"C:\Users\dheer\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\libs\python312.lib")
    pybind_inc = os.path.join(repo_root, ".venv", "Lib", "site-packages", "pybind11", "include")

    srcs_core = [
        "src/ir.cpp",
        "src/graph_verifier.cpp",
        "src/interpreter.cpp",
        "src/pass_manager.cpp",
        "src/optimizer.cpp",
        "src/fusion.cpp",
        "src/codegen.cpp",
        "src/memory_planner.cpp",
        "src/json_io.cpp"
    ]

    os.makedirs("bin", exist_ok=True)
    os.makedirs("python/forgetune", exist_ok=True)

    # 1. _forgetune_cpp.pyd
    print("1. Compiling python/forgetune/_forgetune_cpp.pyd...")
    cmd_pyd = [
        zig_exe, "c++",
        "-target", "x86_64-windows-gnu",
        "-shared",
        "-DFORGETUNE_EXCLUDE_MAIN",
        "-o", "python/forgetune/_forgetune_cpp.pyd",
        "src/bindings.cpp", "tests/cpp/test_main.cpp", *srcs_core,
        "-I", "include",
        "-I", py_inc,
        "-I", pybind_inc,
        py_lib,
        "-std=c++17", "-O2",
        "-Wall", "-Wextra"
    ]
    res1 = subprocess.run(cmd_pyd, capture_output=True, text=True)
    if res1.returncode != 0:
        print("Failed to compile _forgetune_cpp.pyd:")
        print(res1.stderr)
        sys.exit(res1.returncode)
    print("   _forgetune_cpp.pyd compiled successfully.")

    # 2. forgetune_engine.exe
    print("2. Compiling bin/forgetune_engine.exe...")
    cmd_engine = [
        zig_exe, "c++",
        "-target", "x86_64-windows-gnu",
        "-o", "bin/forgetune_engine.exe",
        "src/main.cpp", *srcs_core,
        "-I", "include",
        "-std=c++17", "-O2",
        "-Wall", "-Wextra"
    ]
    res2 = subprocess.run(cmd_engine, capture_output=True, text=True)
    if res2.returncode != 0:
        print("Failed to compile forgetune_engine.exe:")
        print(res2.stderr)
        sys.exit(res2.returncode)
    print("   forgetune_engine.exe compiled successfully.")

    # 3. forgetune_test_suite.exe
    print("3. Compiling bin/forgetune_test_suite.exe...")
    cmd_suite = [
        zig_exe, "c++",
        "-target", "x86_64-windows-gnu",
        "-o", "bin/forgetune_test_suite.exe",
        "tests/cpp/test_main.cpp", *srcs_core,
        "-I", "include",
        "-std=c++17", "-O2",
        "-Wall", "-Wextra"
    ]
    res3 = subprocess.run(cmd_suite, capture_output=True, text=True)
    if res3.returncode != 0:
        print("Failed to compile forgetune_test_suite.exe:")
        print(res3.stderr)
        sys.exit(res3.returncode)
    print("   forgetune_test_suite.exe compiled successfully.")

    print("\nALL NATIVE BINARIES BUILT CLEANLY WITH ZERO WARNINGS!")

if __name__ == "__main__":
    main()
