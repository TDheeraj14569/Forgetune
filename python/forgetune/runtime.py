"""
Native shared-library loader and runtime execution engine for ForgeTune.
Binds to compiled C++ kernels using ctypes with contract validation and safe staging.
"""

import os
import sys
import shutil
import ctypes
import subprocess
import tempfile
import time
import numpy as np
from typing import Tuple, Optional
from .errors import RuntimeError, InvalidShapeError

class NativeRuntime:
    def __init__(self, dll_path: str, expected_input_shape: Tuple[int, ...], expected_output_shape: Tuple[int, ...]):
        self.original_dll_path = os.path.abspath(dll_path)
        self.expected_input_shape = tuple(expected_input_shape)
        self.expected_output_shape = tuple(expected_output_shape)
        self.total_input_count = int(np.prod(self.expected_input_shape))
        self.total_output_count = int(np.prod(self.expected_output_shape))

        self.lib = self._load_library()
        self._init_entry_point()

    def _load_library(self) -> ctypes.CDLL:
        if not os.path.isfile(self.original_dll_path):
            raise FileNotFoundError(f"Shared library not found: {self.original_dll_path}")

        # On Windows, always copy the DLL to a unique temporary file in %TEMP%
        # to prevent Windows from locking the build artifact on disk and to comply with WDAC policies.
        if sys.platform == "win32":
            temp_dir = os.path.join(tempfile.gettempdir(), "forgetune_runtime")
            os.makedirs(temp_dir, exist_ok=True)
            import uuid
            unique_name = f"{os.path.splitext(os.path.basename(self.original_dll_path))[0]}_{uuid.uuid4().hex[:8]}.dll"
            staged_path = os.path.join(temp_dir, unique_name)
            shutil.copy2(self.original_dll_path, staged_path)
            # Remove Zone.Identifier NTFS alternate stream if present to unblock without slow PowerShell process spawn
            try:
                zone_path = staged_path + ":Zone.Identifier"
                if os.path.exists(zone_path):
                    os.remove(zone_path)
            except Exception:
                pass

            # Windows Defender / SmartScreen evaluates newly created binaries asynchronously.
            # If WinError 4551 (STATUS_NEEDS_REMEDIATION) occurs, retry with backoff.
            for attempt in range(30):
                try:
                    return ctypes.CDLL(staged_path)
                except OSError as e:
                    if "4551" in str(e) and attempt < 29:
                        if attempt == 2 or attempt == 10:
                            try:
                                subprocess.run(
                                    ["powershell", "-NoProfile", "-Command", f"Unblock-File -LiteralPath '{staged_path}'"],
                                    capture_output=True,
                                    timeout=3
                                )
                            except Exception:
                                pass
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    # Fallback: try loading the original DLL directly
                    try:
                        return ctypes.CDLL(self.original_dll_path)
                    except Exception:
                        pass
                    raise
            return ctypes.CDLL(staged_path)
        else:
            return ctypes.CDLL(self.original_dll_path)

    def _init_entry_point(self) -> None:
        if not hasattr(self.lib, "forgetune_run"):
            raise RuntimeError(
                f"Missing exported entry point 'forgetune_run' in {self.original_dll_path}"
            )

        self.run_fn = self.lib.forgetune_run
        self.run_fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input buffer
            ctypes.c_size_t,                 # input element count
            ctypes.POINTER(ctypes.c_float),  # output buffer
            ctypes.c_size_t                  # output element count
        ]
        self.run_fn.restype = ctypes.c_int

        if hasattr(self.lib, "forgetune_benchmark_kernel"):
            self.benchmark_fn = self.lib.forgetune_benchmark_kernel
            self.benchmark_fn.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_int
            ]
            self.benchmark_fn.restype = ctypes.c_double
        else:
            self.benchmark_fn = None

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Executes native inference on the given input tensor.
        Validates shape, dtype, and status code.
        """
        # Validate shape
        if tuple(input_tensor.shape) != self.expected_input_shape:
            raise InvalidShapeError(
                f"Input shape mismatch: expected {self.expected_input_shape}, got {input_tensor.shape}"
            )

        # Ensure contiguous float32
        if input_tensor.dtype != np.float32 or not input_tensor.flags["C_CONTIGUOUS"]:
            input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)

        # Allocate output buffer
        output_tensor = np.zeros(self.expected_output_shape, dtype=np.float32)

        inp_ptr = input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = output_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Execute native function
        status = self.run_fn(
            inp_ptr,
            self.total_input_count,
            out_ptr,
            self.total_output_count
        )

        if status != 0:
            error_msgs = {
                -1: "Null pointer supplied to native kernel",
                -2: f"Input element count mismatch (expected {self.total_input_count})",
                -3: f"Output element count mismatch (expected {self.total_output_count})"
            }
            msg = error_msgs.get(status, f"Unknown native error code: {status}")
            raise RuntimeError(f"Native kernel execution failed with code {status}: {msg}")

        return output_tensor

    def benchmark_kernel(self, input_tensor: np.ndarray, iters: int = 1000) -> float:
        """
        Executes the in-kernel benchmark inside pure C++ without ctypes per-call overhead.
        Returns average kernel latency in microseconds.
        """
        if self.benchmark_fn is None:
            raise RuntimeError("DLL does not export forgetune_benchmark_kernel")

        if tuple(input_tensor.shape) != self.expected_input_shape:
            raise InvalidShapeError(
                f"Input shape mismatch: expected {self.expected_input_shape}, got {input_tensor.shape}"
            )
        if input_tensor.dtype != np.float32 or not input_tensor.flags["C_CONTIGUOUS"]:
            input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)

        output_tensor = np.zeros(self.expected_output_shape, dtype=np.float32)
        inp_ptr = input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = output_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        lat_us = self.benchmark_fn(
            inp_ptr,
            self.total_input_count,
            out_ptr,
            self.total_output_count,
            iters
        )
        if lat_us < 0:
            raise RuntimeError(f"Native benchmark returned error code: {lat_us}")
        return float(lat_us)
