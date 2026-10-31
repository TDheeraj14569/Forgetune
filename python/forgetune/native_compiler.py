"""
Native C++ compiler driver for ForgeTune.
Invokes detected native compilers (Zig C++, Clang, GCC, MSVC) with explicit argument lists,
shell execution disabled, timeout protection, and artifact staging.
"""

import os
import sys
import shutil
import subprocess
import time
import uuid
import platform
from typing import Dict, Any, Optional
from .errors import CompilationError

class NativeCompiler:
    def __init__(self, compiler_path: Optional[str] = None):
        self.compiler_path = compiler_path or self._detect_compiler()
        self.compiler_type = self._detect_compiler_type(self.compiler_path)

    def _detect_compiler(self) -> str:
        # Check 1: Zig C++ in virtual environment (guaranteed working C++17 on Windows)
        venv_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
        zig_pkg = os.path.join(venv_dir, "Lib", "site-packages", "ziglang", "zig.exe")
        if os.path.isfile(zig_pkg):
            return zig_pkg

        # Check 2: clang++ in PATH
        clang_path = shutil.which("clang++")
        if clang_path:
            return clang_path

        # Check 3: g++ in PATH
        gxx_path = shutil.which("g++")
        if gxx_path:
            return gxx_path

        # Check 4: cl in PATH
        cl_path = shutil.which("cl")
        if cl_path:
            return cl_path

        # Fallback to zig in Lib
        return "zig"

    def _detect_compiler_type(self, path: str) -> str:
        base = os.path.basename(path).lower()
        if "zig" in base:
            return "zig"
        elif "clang" in base:
            return "clang"
        elif "g++" in base or "gcc" in base:
            return "gcc"
        elif base in ("cl", "cl.exe"):
            return "msvc"
        return "unknown"

    def get_compiler_info(self) -> Dict[str, Any]:
        info = {
            "path": self.compiler_path,
            "type": self.compiler_type,
            "version": "unknown"
        }
        try:
            if self.compiler_type == "zig":
                cmd = [self.compiler_path, "version"]
            elif self.compiler_type in ("clang", "gcc"):
                cmd = [self.compiler_path, "--version"]
            elif self.compiler_type == "msvc":
                cmd = [self.compiler_path]
            else:
                cmd = [self.compiler_path, "--version"]

            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            ver = (res.stdout or res.stderr).splitlines()
            if ver:
                info["version"] = ver[0].strip()
        except Exception as e:
            info["version"] = f"detection failed: {e}"
        return info

    def compile_shared_library(
        self,
        cpp_source_path: str,
        output_dll_path: str,
        opt_level: str = "-O2",
        timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Compiles a C++ source file into a shared library.
        Returns a dict containing compilation metadata (duration, command, stdout, stderr).
        """
        # Canonical optimization level normalization
        CANONICAL_MAP = {
            "-O0": "-O0", "-O1": "-O1", "-O2": "-O2", "-O3": "-O3", "-Os": "-Os", "-Oz": "-Oz",
            "/Od": "-O0", "/O1": "-O1", "/O2": "-O2", "/Ox": "-O3",
            "O0": "-O0", "O1": "-O1", "O2": "-O2", "O3": "-O3", "Os": "-Os", "Oz": "-Oz",
            "Od": "-O0", "Ox": "-O3"
        }
        if opt_level not in CANONICAL_MAP:
            raise ValueError(f"Invalid optimization level '{opt_level}'. Allowed levels: -O0, -O1, -O2, -O3, -Os, -Oz")
        canonical_opt = CANONICAL_MAP[opt_level]

        if not os.path.isfile(cpp_source_path):
            raise FileNotFoundError(f"C++ source file not found: {cpp_source_path}")

        dest_dir = os.path.dirname(os.path.abspath(output_dll_path))
        os.makedirs(dest_dir, exist_ok=True)

        # Detect and validate machine architecture
        raw_arch = platform.machine().lower()
        if raw_arch in ("x86_64", "amd64"):
            target_arch = "x86_64"
        elif raw_arch in ("aarch64", "arm64"):
            target_arch = "aarch64"
        else:
            raise CompilationError(
                f"Unsupported system architecture '{platform.machine()}': ForgeTune native compiler requires x86_64 or aarch64."
            )

        # Stage to a unique collision-resistant temporary file
        temp_dll_path = os.path.join(
            dest_dir,
            f".tmp_{os.getpid()}_{uuid.uuid4().hex}_{os.path.basename(output_dll_path)}"
        )
        if os.path.isfile(temp_dll_path):
            try:
                os.remove(temp_dll_path)
            except OSError:
                pass

        if self.compiler_type == "zig":
            target_triple = f"{target_arch}-windows-gnu" if sys.platform == "win32" else f"{target_arch}-linux-gnu"
            target_flag = ["-target", target_triple]
            if sys.platform != "win32":
                target_flag.append("-fPIC")
            cmd = [
                self.compiler_path,
                "c++",
                *target_flag,
                "-shared",
                "-o", temp_dll_path,
                cpp_source_path,
                "-std=c++17",
                canonical_opt,
                "-Wall",
                "-Wextra",
                "-Wno-nullability-completeness",
                "-fno-signed-zeros"
            ]
        elif self.compiler_type in ("clang", "gcc"):
            cmd = [
                self.compiler_path,
                "-shared",
                "-fPIC",
                "-o", temp_dll_path,
                cpp_source_path,
                "-std=c++17",
                canonical_opt,
                "-Wall",
                "-Wextra",
                "-fno-signed-zeros"
            ]
            if self.compiler_type == "clang":
                cmd.append("-Wno-nullability-completeness")
        elif self.compiler_type == "msvc":
            msvc_opt_map = {"-O0": "/Od", "-O1": "/O1", "-O2": "/O2", "-O3": "/O2", "-Os": "/O1", "-Oz": "/O1"}
            msvc_opt = msvc_opt_map.get(canonical_opt, "/O2")
            cmd = [
                self.compiler_path,
                "/LD",
                cpp_source_path,
                f"/Fe:{temp_dll_path}",
                "/std:c++17",
                msvc_opt,
                "/fp:precise",
                "/W4"
            ]
        else:
            cmd = [
                self.compiler_path,
                "-shared",
                "-o", temp_dll_path,
                cpp_source_path,
                "-std=c++17",
                canonical_opt
            ]

        start_time = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                shell=False
            )
        except subprocess.TimeoutExpired as e:
            if os.path.isfile(temp_dll_path):
                try:
                    os.remove(temp_dll_path)
                except OSError:
                    pass
            raise CompilationError(
                f"Compilation timed out after {timeout_seconds} seconds",
                command=cmd
            ) from e
        except Exception as e:
            if os.path.isfile(temp_dll_path):
                try:
                    os.remove(temp_dll_path)
                except OSError:
                    pass
            raise CompilationError(
                f"Failed to execute compiler: {e}",
                command=cmd
            ) from e

        duration_sec = time.perf_counter() - start_time

        if result.returncode != 0 or not os.path.isfile(temp_dll_path):
            if os.path.isfile(temp_dll_path):
                try:
                    os.remove(temp_dll_path)
                except OSError:
                    pass
            raise CompilationError(
                f"Compilation failed with exit code {result.returncode}",
                stdout=result.stdout,
                stderr=result.stderr,
                command=cmd
            )

        # Atomically replace final DLL path; catch locked DLL on Windows
        try:
            os.replace(temp_dll_path, output_dll_path)
        except (PermissionError, OSError) as e:
            if os.path.isfile(temp_dll_path):
                try:
                    os.remove(temp_dll_path)
                except OSError:
                    pass
            raise CompilationError(
                f"Failed to overwrite shared library '{output_dll_path}': file is locked by an active process or runtime. "
                f"Ensure all NativeRuntime / CompiledModel instances using this library are closed before recompiling.",
                command=cmd
            ) from e

        return {
            "success": True,
            "command": cmd,
            "duration_seconds": duration_sec,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output_path": output_dll_path,
            "size_bytes": os.path.getsize(output_dll_path)
        }
