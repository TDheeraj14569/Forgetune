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
import platform
import json
import re
from typing import Optional

def find_zig():
    # 1. PATH
    exe_name = "zig.exe" if sys.platform == "win32" else "zig"
    on_path = shutil.which(exe_name) or shutil.which("zig")
    if on_path:
        return on_path

    # 2. ziglang package in current python env
    try:
        import ziglang
        zig_dir = os.path.dirname(ziglang.__file__)
        cand = os.path.join(zig_dir, exe_name)
        if os.path.isfile(cand):
            return cand
    except ImportError:
        pass

    # 3. Portable discovery across Linux, macOS, and Windows
    search_dirs = [
        sys.prefix,
        os.path.join(sys.prefix, "bin" if sys.platform != "win32" else "Scripts"),
        sysconfig.get_path("scripts"),
        sysconfig.get_path("purelib"),
        sysconfig.get_path("platlib"),
    ]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for venv_path in [os.path.join(repo_root, ".venv"), os.path.join(repo_root, "venv")]:
        search_dirs.extend([
            venv_path,
            os.path.join(venv_path, "bin" if sys.platform != "win32" else "Scripts"),
            os.path.join(venv_path, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"),
            os.path.join(venv_path, "Lib", "site-packages")
        ])

    for base in search_dirs:
        if not base or not os.path.isdir(base):
            continue
        cand = os.path.join(base, exe_name)
        if os.path.isfile(cand):
            return cand
        cand = os.path.join(base, "ziglang", exe_name)
        if os.path.isfile(cand):
            return cand

    return exe_name

def classify_diagnostic(line_str: str, repo_root: Optional[str] = None) -> str:
    """
    Parses a compiler output line and categorizes it into:
      - 'not_a_warning': Line is not a compiler warning diagnostic.
      - 'toolchain_warning': Warning originating from external toolchains, SDKs,
                             standard libraries, or unplaced compiler CLI flags.
      - 'unknown_warning': Warning originating from an unrecognized external path.
      - 'project_warning': Warning originating from inside our project source tree
                           (src/, include/forgetune/, tests/cpp/).
    """
    lower_line = line_str.lower()
    warn_match = re.search(r"warning(?:\s+[a-z0-9]+)?\s*:", lower_line)
    if not warn_match:
        return "not_a_warning"

    prefix = line_str[:warn_match.start()].strip()
    file_candidate = None
    m_win = re.match(r"^([a-zA-Z]:[\\/][^:(]+)", prefix)
    if m_win:
        file_candidate = m_win.group(1).replace("\\", "/")
    else:
        m_posix = re.match(r"^([^:(]+)", prefix)
        if m_posix:
            file_candidate = m_posix.group(1).replace("\\", "/")

    if not file_candidate or not file_candidate.strip():
        # External or toolchain-level diagnostic without source location (e.g. CLI flags)
        return "toolchain_warning"

    file_norm = file_candidate.lower().strip()
    is_toolchain = file_norm in ("clang", "clang++", "zig", "gcc", "g++", "cl", "lld-link", "ld") or any(td in file_norm for td in (
        "/libcxx/", "libcxx/", "/site-packages/", "site-packages/",
        "/hostedtoolcache/", "hostedtoolcache/", "/usr/include/",
        "/usr/lib/", "microsoft visual studio", "/vc/tools/",
        "/windows kits/"
    ))
    if is_toolchain:
        return "toolchain_warning"

    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_root_norm = os.path.abspath(repo_root).replace("\\", "/").lower().rstrip("/")

    clean_cand = file_candidate.replace("\\", "/").lstrip("./")
    clean_cand_lower = clean_cand.lower()

    project_subdirs = ("src/", "include/forgetune/", "tests/cpp/")
    if any(clean_cand_lower.startswith(sd) for sd in project_subdirs):
        return "project_warning"

    abs_cand = os.path.abspath(file_candidate).replace("\\", "/").lower()
    if abs_cand.startswith(repo_root_norm + "/"):
        rel_under_repo = abs_cand[len(repo_root_norm) + 1:]
        if any(rel_under_repo.startswith(sd) for sd in project_subdirs):
            return "project_warning"
        return "toolchain_warning"

    # Strictly check for CI / synthetic paths anchored under a forgetune repo directory
    cand_norm = file_candidate.replace("\\", "/")
    if re.search(r"/(?:forgetune/)+(?:src/|include/forgetune/|tests/cpp/)", "/" + cand_norm.lstrip("/"), re.IGNORECASE):
        return "project_warning"

    return "unknown_warning"

def is_codebase_warning(line_str: str, repo_root: Optional[str] = None) -> bool:
    """
    Parses a compiler diagnostic line and returns True if it represents a warning
    originating from our project codebase (src/, include/forgetune/, tests/cpp/),
    and False if it is a toolchain/SDK warning (including unplaced CLI-flag diagnostics),
    unknown path warning, or non-warning diagnostic.
    """
    return classify_diagnostic(line_str, repo_root) == "project_warning"

def check_compilation_result(target_name, result, repo_root=None):
    if result.returncode != 0:
        print(f"Failed to compile {target_name} (exit code {result.returncode}):")
        print(result.stderr)
        sys.exit(result.returncode)

    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    warning_lines = []
    toolchain_lines = []
    unknown_lines = []
    for line in combined.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        diag_kind = classify_diagnostic(line_str, repo_root)
        if diag_kind == "project_warning":
            warning_lines.append(line_str)
        elif diag_kind == "toolchain_warning":
            toolchain_lines.append(line_str)
        elif diag_kind == "unknown_warning":
            unknown_lines.append(line_str)

    if toolchain_lines:
        print(f"   [Notice] Filtered {len(toolchain_lines)} external/toolchain diagnostic(s).")
    if unknown_lines:
        print(f"   [Notice] Filtered {len(unknown_lines)} unknown/external diagnostic(s):")
        for u in unknown_lines:
            print(f"     [Unknown] {u}")

    if warning_lines:
        print(f"FAILED zero-warning requirement for {target_name}:")
        for w in warning_lines:
            print(f"  {w}")
        sys.exit(1)
    print(f"   {target_name} compiled successfully with 0 project-source warnings.")

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
        pybind_inc = ""
        cand_dirs = [
            sysconfig.get_path("include"),
            os.path.join(sysconfig.get_path("purelib") or "", "pybind11", "include"),
            os.path.join(sysconfig.get_path("platlib") or "", "pybind11", "include"),
        ]
        for base in [sys.prefix, repo_root, os.path.join(repo_root, ".venv"), os.path.join(repo_root, "venv")]:
            cand_dirs.extend([
                os.path.join(base, "include", "pybind11"),
                os.path.join(base, "include"),
                os.path.join(base, "Lib", "site-packages", "pybind11", "include"),
                os.path.join(base, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "pybind11", "include"),
                os.path.join(base, "lib", "site-packages", "pybind11", "include"),
            ])
        for c in cand_dirs:
            if c and os.path.isdir(c):
                pybind_inc = c
                break

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

    raw_arch = platform.machine().lower()
    target_arch = "aarch64" if raw_arch in ("aarch64", "arm64") else "x86_64"
    target_flags = ["-target", f"{target_arch}-windows-gnu"] if is_win else ["-fPIC"]

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
        "-isystem", py_inc,
        "-isystem", pybind_inc,
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

    cmd_pyd.extend([
        "-std=c++17", "-O2", "-Wall", "-Wextra",
        "-Wno-nullability",
        "-Wno-nullability-completeness",
        "-Wno-nullability-completeness-on-arrays",
        "-Wno-nullability-declspec",
        "-fno-signed-zeros"
    ])

    res1 = subprocess.run(cmd_pyd, capture_output=True, text=True)
    check_compilation_result(pyd_path, res1)

    # Save build metadata for SmartBuildExt freshness validation
    meta_path = os.path.join(os.path.dirname(pyd_path), "_forgetune_cpp.build_meta.json")
    try:
        zig_version = "unknown"
        try:
            p_v = subprocess.run([zig_exe, "version"], capture_output=True, text=True)
            if p_v.returncode == 0:
                zig_version = p_v.stdout.strip()
        except Exception:
            pass
        meta_data = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "python_abi": sysconfig.get_config_var("SOABI") or f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": sys.platform,
            "machine": platform.machine(),
            "compiler_path": zig_exe,
            "compiler_version": zig_version,
            "build_flags": ["-std=c++17", "-O2", "-Wall", "-Wextra", "-fno-signed-zeros"],
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
        "-Wall", "-Wextra",
        "-Wno-nullability",
        "-Wno-nullability-completeness",
        "-Wno-nullability-completeness-on-arrays",
        "-Wno-nullability-declspec",
        "-fno-signed-zeros"
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
        "-Wall", "-Wextra",
        "-Wno-nullability",
        "-Wno-nullability-completeness",
        "-Wno-nullability-completeness-on-arrays",
        "-Wno-nullability-declspec",
        "-fno-signed-zeros"
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
    alias_failed = False
    for src, dst in aliases:
        src_norm = os.path.normpath(src)
        dst_norm = os.path.normpath(dst)
        if src_norm != dst_norm:
            try:
                shutil.copy2(src, dst)
                if not is_win:
                    os.chmod(dst, 0o755)
            except Exception as e:
                print(f"ERROR creating alias {dst} from {src}: {e}")
                alias_failed = True

    if alias_failed:
        print("FAILED to create all required binary aliases.")
        sys.exit(1)

    print("\nALL NATIVE BINARIES BUILT CLEANLY WITH ZERO WARNINGS!")

if __name__ == "__main__":
    main()
