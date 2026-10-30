"""
Build script to compile all native ForgeTune C++ targets using Zig.
Targets:
1. python/forgetune/_forgetune_cpp.pyd (or .so on POSIX) (Pybind11 extension)
2. bin/forgetune_engine.exe (or bin/forgetune_engine on POSIX) (Compiler engine CLI)
3. bin/forgetune_test_suite.exe (or bin/forgetune_test_suite on POSIX) (C++ native test suite)
"""

import os
import sys
import shutil
import subprocess
import sysconfig

def find_zig():
    # 1. PATH
    on_path = shutil.which("zig")
    if on_path:
        return on_path

    # 2. ziglang package in current python env
    try:
        import ziglang
        zig_dir = os.path.dirname(ziglang.__file__)
        exe_name = "zig.exe" if sys.platform == "win32" else "zig"
        cand = os.path.join(zig_dir, exe_name)
        if os.path.isfile(cand):
            return cand
    except ImportError:
        pass

    # 3. repo root .venv or common virtualenv locations
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for base in [sys.prefix, repo_root, os.path.join(repo_root, ".venv")]:
        for sub in [
            ("Lib", "site-packages", "ziglang", "zig.exe"),
            ("lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "ziglang", "zig"),
            ("Scripts", "zig.exe"),
            ("bin", "zig")
        ]:
            cand = os.path.join(base, *sub)
            if os.path.isfile(cand):
                return cand

    return "zig"

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    zig_exe = find_zig()

    # Portable Python include path discovery
    py_inc = sysconfig.get_path("include")
    if not py_inc or not os.path.isdir(py_inc):
        py_inc = sysconfig.get_paths().get("include", "")

    # Portable Pybind11 include path discovery
    try:
        import pybind11
        pybind_inc = pybind11.get_include()
    except ImportError:
        pybind_inc = os.path.join(sys.prefix, "Lib", "site-packages", "pybind11", "include")
        if not os.path.isdir(pybind_inc):
            pybind_inc = os.path.join(repo_root, ".venv", "Lib", "site-packages", "pybind11", "include")

    is_win = sys.platform == "win32"
    pyd_ext = ".pyd" if is_win else ".so"
    exe_ext = ".exe" if is_win else ""

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

    target_flags = ["-target", "x86_64-windows-gnu"] if is_win else ["-fPIC"]

    # 1. _forgetune_cpp extension
    pyd_path = f"python/forgetune/_forgetune_cpp{pyd_ext}"
    print(f"1. Compiling {pyd_path}...")
    cmd_pyd = [
        zig_exe, "c++",
        *target_flags,
        "-shared",
        "-DFORGETUNE_EXCLUDE_MAIN",
        "-o", pyd_path,
        "src/bindings.cpp", "tests/cpp/test_main.cpp", *srcs_core,
        "-I", "include",
        "-I", py_inc,
        "-I", pybind_inc,
    ]

    if is_win:
        # Dynamically discover pythonXY.lib on Windows
        lib_name = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
        py_lib = None
        cand_dirs = [
            os.path.join(sys.base_prefix, "libs"),
            os.path.join(sys.prefix, "libs"),
        ]
        inst_base = sysconfig.get_config_var("installed_base")
        if inst_base:
            cand_dirs.append(os.path.join(inst_base, "libs"))

        for d in cand_dirs:
            if d and os.path.isfile(os.path.join(d, lib_name)):
                py_lib = os.path.join(d, lib_name)
                break

        if not py_lib:
            for root, _, files in os.walk(sys.base_prefix):
                if lib_name in files:
                    py_lib = os.path.join(root, lib_name)
                    break

        if py_lib:
            cmd_pyd.append(py_lib)
        else:
            raise RuntimeError(f"Could not locate Windows Python import library '{lib_name}'")

    cmd_pyd.extend(["-std=c++17", "-O2", "-Wall", "-Wextra"])

    def check_compilation_result(target_name, result):
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        warning_lines = []
        for line in combined.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            if "warning:" in line_str.lower():
                warning_lines.append(line_str)
        if result.returncode != 0:
            print(f"Failed to compile {target_name} (exit code {result.returncode}):")
            print(result.stderr)
            sys.exit(result.returncode)
        if warning_lines:
            print(f"FAILED zero-warning requirement for {target_name}:")
            for w in warning_lines:
                print(f"  {w}")
            sys.exit(1)
        print(f"   {target_name} compiled successfully with 0 warnings.")

    res1 = subprocess.run(cmd_pyd, capture_output=True, text=True)
    check_compilation_result(pyd_path, res1)

    # Save build metadata for SmartBuildExt freshness validation
    meta_path = os.path.join(os.path.dirname(pyd_path), "_forgetune_cpp.build_meta.json")
    try:
        import json, platform
        meta_data = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "python_abi": sysconfig.get_config_var("SOABI") or f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": sys.platform,
            "machine": platform.machine(),
            "compiler_path": zig_exe,
            "build_flags": ["-std=c++17", "-O2", "-Wall", "-Wextra"],
            "pybind11_version": getattr(pybind11, "__version__", "unknown")
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
    except Exception as e:
        print(f"   Notice: could not write build metadata ({e})")

    # 2. forgetune_engine executable
    engine_path = f"bin/forgetune_engine{exe_ext}"
    print(f"2. Compiling {engine_path}...")
    cmd_engine = [
        zig_exe, "c++",
        *target_flags,
        "-o", engine_path,
        "src/main.cpp", *srcs_core,
        "-I", "include",
        "-std=c++17", "-O2",
        "-Wall", "-Wextra"
    ]
    res2 = subprocess.run(cmd_engine, capture_output=True, text=True)
    check_compilation_result(engine_path, res2)

    # 3. forgetune_test_suite executable
    suite_path = f"bin/forgetune_test_suite{exe_ext}"
    print(f"3. Compiling {suite_path}...")
    cmd_suite = [
        zig_exe, "c++",
        *target_flags,
        "-o", suite_path,
        "tests/cpp/test_main.cpp", *srcs_core,
        "-I", "include",
        "-std=c++17", "-O2",
        "-Wall", "-Wextra"
    ]
    res3 = subprocess.run(cmd_suite, capture_output=True, text=True)
    check_compilation_result(suite_path, res3)

    # Cross-platform compatibility: ensure both .exe and extensionless binaries
    # exist, as well as forgetune_test_engine aliases, so both Windows and Linux CI
    # and all test harnesses find the executable immediately.
    aliases = [
        (engine_path, "bin/forgetune_engine.exe"),
        (engine_path, "bin/forgetune_engine"),
        (engine_path, "bin/forgetune_test_engine.exe"),
        (engine_path, "bin/forgetune_test_engine"),
        (suite_path, "bin/forgetune_test_suite.exe"),
        (suite_path, "bin/forgetune_test_suite"),
    ]
    for src, dst in aliases:
        src_norm = os.path.normpath(src)
        dst_norm = os.path.normpath(dst)
        if src_norm != dst_norm:
            try:
                shutil.copy2(src, dst)
                if not is_win:
                    os.chmod(dst, 0o755)
            except Exception as e:
                print(f"Warning creating alias {dst} from {src}: {e}")

    print("\nALL NATIVE BINARIES BUILT CLEANLY WITH ZERO WARNINGS!")

if __name__ == "__main__":
    main()
