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
from typing import Dict, Any, List, Optional
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
        elif "cl" in base:
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
        if not os.path.isfile(cpp_source_path):
            raise FileNotFoundError(f"C++ source file not found: {cpp_source_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_dll_path)), exist_ok=True)

        if self.compiler_type == "zig":
            # zig c++ -target x86_64-windows-gnu -shared -o <dll> <src> -std=c++17 <opt_level> -w
            cmd = [
                self.compiler_path,
                "c++",
                "-target", "x86_64-windows-gnu",
                "-shared",
                "-o", output_dll_path,
                cpp_source_path,
                "-std=c++17",
                opt_level,
                "-w"
            ]
        elif self.compiler_type in ("clang", "gcc"):
            cmd = [
                self.compiler_path,
                "-shared",
                "-fPIC",
                "-o", output_dll_path,
                cpp_source_path,
                "-std=c++17",
                opt_level,
                "-Wall"
            ]
        elif self.compiler_type == "msvc":
            cmd = [
                self.compiler_path,
                "/LD",
                cpp_source_path,
                f"/Fe:{output_dll_path}",
                "/std:c++17",
                opt_level.replace("-O", "/O")
            ]
        else:
            cmd = [
                self.compiler_path,
                "-shared",
                "-o", output_dll_path,
                cpp_source_path,
                "-std=c++17",
                opt_level
            ]

        start_time = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                shell=False
            )
        except subprocess.TimeoutExpired as e:
            raise CompilationError(
                f"Compilation timed out after {timeout_seconds} seconds",
                command=cmd
            ) from e
        except Exception as e:
            raise CompilationError(
                f"Failed to execute compiler: {e}",
                command=cmd
            ) from e

        duration_sec = time.perf_counter() - start_time

        if result.returncode != 0 or not os.path.isfile(output_dll_path):
            raise CompilationError(
                f"Compilation failed with exit code {result.returncode}",
                stdout=result.stdout,
                stderr=result.stderr,
                command=cmd
            )

        # On Windows, unblock the file if PowerShell is available
        if sys.platform == "win32":
            try:
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", f"Unblock-File -LiteralPath '{output_dll_path}'"],
                    capture_output=True,
                    timeout=5,
                    shell=False
                )
            except Exception:
                pass

        return {
            "success": True,
            "command": cmd,
            "duration_seconds": duration_sec,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output_path": output_dll_path,
            "size_bytes": os.path.getsize(output_dll_path)
        }
