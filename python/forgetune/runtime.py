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
        self.is_shape_dynamic = (len(self.expected_input_shape) > 0 and self.expected_input_shape[0] == -1)
        if self.is_shape_dynamic:
            self.total_input_count = -1
            self.total_output_count = -1
        else:
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
            unique_name = f"forgetune_test_{os.path.splitext(os.path.basename(self.original_dll_path))[0]}_{uuid.uuid4().hex[:8]}.dll"
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

        if hasattr(self.lib, "forgetune_is_dynamic"):
            self.is_dynamic = bool(self.lib.forgetune_is_dynamic())
        elif hasattr(self.lib, "forgetune_run_v1"):
            self.is_dynamic = True
        else:
            self.is_dynamic = self.is_shape_dynamic

        if hasattr(self.lib, "forgetune_max_batch"):
            self.max_batch_size = int(self.lib.forgetune_max_batch())
        else:
            self.max_batch_size = 128 if self.is_dynamic else (self.expected_input_shape[0] if len(self.expected_input_shape) > 0 else 1)

        if self.is_dynamic:
            self.run_dynamic_fn = self.lib.forgetune_run
            self.run_dynamic_fn.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_int64
            ]
            self.run_dynamic_fn.restype = ctypes.c_int
            self.run_fn = self.run_dynamic_fn

            if hasattr(self.lib, "forgetune_run_v2"):
                self.run_v2_fn = self.lib.forgetune_run_v2
                self.run_v2_fn.argtypes = self.run_dynamic_fn.argtypes
                self.run_v2_fn.restype = ctypes.c_int
            else:
                self.run_v2_fn = self.run_dynamic_fn

            if hasattr(self.lib, "forgetune_run_v1"):
                self.run_v1_fn = self.lib.forgetune_run_v1
                self.run_v1_fn.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_size_t
                ]
                self.run_v1_fn.restype = ctypes.c_int
        else:
            self.run_static_fn = self.lib.forgetune_run
            self.run_static_fn.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t
            ]
            self.run_static_fn.restype = ctypes.c_int
            self.run_fn = self.run_static_fn

            if hasattr(self.lib, "forgetune_run_v1"):
                self.run_v1_fn = self.lib.forgetune_run_v1
                self.run_v1_fn.argtypes = self.run_static_fn.argtypes
                self.run_v1_fn.restype = ctypes.c_int
            else:
                self.run_v1_fn = self.run_static_fn

            if hasattr(self.lib, "forgetune_run_v2"):
                self.run_v2_fn = self.lib.forgetune_run_v2
                self.run_v2_fn.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_size_t,
                    ctypes.c_int64
                ]
                self.run_v2_fn.restype = ctypes.c_int

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

    def run(self, input_tensor: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Executes native inference on the given input tensor.
        Validates shape, dtype, and status code.
        """
        if self.is_dynamic:
            # Validate rank
            if input_tensor.ndim != len(self.expected_input_shape):
                raise InvalidShapeError(
                    f"Input rank mismatch: expected {len(self.expected_input_shape)}D, got {input_tensor.ndim}D"
                )
            # Validate non-batch dimensions
            for d in range(1, len(self.expected_input_shape)):
                if self.expected_input_shape[d] != -1 and input_tensor.shape[d] != self.expected_input_shape[d]:
                    raise InvalidShapeError(
                        f"Input dimension {d} mismatch: expected {self.expected_input_shape[d]}, got {input_tensor.shape[d]}"
                    )

            actual_batch = input_tensor.shape[0]
            if batch_size is not None and batch_size != actual_batch:
                raise InvalidShapeError(
                    f"Explicit batch_size {batch_size} does not match input leading dimension {actual_batch}"
                )
            if actual_batch <= 0 or actual_batch > self.max_batch_size:
                raise InvalidShapeError(
                    f"Dynamic batch size {actual_batch} out of valid range [1, {self.max_batch_size}]"
                )

            # Ensure contiguous float32
            if input_tensor.dtype != np.float32 or not input_tensor.flags["C_CONTIGUOUS"]:
                input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)

            output_shape = (actual_batch, *self.expected_output_shape[1:])
            output_tensor = np.empty(output_shape, dtype=np.float32)

            inp_ptr = input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            out_ptr = output_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            status = self.run_dynamic_fn(
                inp_ptr,
                int(input_tensor.size),
                out_ptr,
                int(output_tensor.size),
                actual_batch
            )

            if status != 0:
                error_msgs = {
                    -1: "Null pointer supplied to native kernel",
                    -2: f"Input element count or batch size mismatch (batch={actual_batch})",
                    -3: f"Output element count mismatch (batch={actual_batch})"
                }
                msg = error_msgs.get(status, f"Unknown native error code: {status}")
                raise RuntimeError(f"Native kernel execution failed with code {status}: {msg}")

            return output_tensor

        else:
            # Validate shape
            if tuple(input_tensor.shape) != self.expected_input_shape:
                raise InvalidShapeError(
                    f"Input shape mismatch: expected {self.expected_input_shape}, got {input_tensor.shape}"
                )

            # Ensure contiguous float32
            if input_tensor.dtype != np.float32 or not input_tensor.flags["C_CONTIGUOUS"]:
                input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)

            # Allocate output buffer
            output_tensor = np.empty(self.expected_output_shape, dtype=np.float32)

            inp_ptr = input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            out_ptr = output_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            # Execute native function
            status = self.run_static_fn(
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

        if self.is_dynamic:
            if input_tensor.ndim != len(self.expected_input_shape):
                raise InvalidShapeError(
                    f"Input rank mismatch: expected {len(self.expected_input_shape)}D, got {input_tensor.ndim}D"
                )
            for d in range(1, len(self.expected_input_shape)):
                if self.expected_input_shape[d] != -1 and input_tensor.shape[d] != self.expected_input_shape[d]:
                    raise InvalidShapeError(
                        f"Input dimension {d} mismatch: expected {self.expected_input_shape[d]}, got {input_tensor.shape[d]}"
                    )
            actual_batch = input_tensor.shape[0]
            if actual_batch <= 0 or actual_batch > self.max_batch_size:
                raise InvalidShapeError(
                    f"Dynamic batch size {actual_batch} out of valid range [1, {self.max_batch_size}]"
                )
            output_shape = (actual_batch, *self.expected_output_shape[1:])
        else:
            if tuple(input_tensor.shape) != self.expected_input_shape:
                raise InvalidShapeError(
                    f"Input shape mismatch: expected {self.expected_input_shape}, got {input_tensor.shape}"
                )
            output_shape = self.expected_output_shape

        if input_tensor.dtype != np.float32 or not input_tensor.flags["C_CONTIGUOUS"]:
            input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)

        output_tensor = np.empty(output_shape, dtype=np.float32)
        inp_ptr = input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = output_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        lat_us = self.benchmark_fn(
            inp_ptr,
            int(input_tensor.size),
            out_ptr,
            int(output_tensor.size),
            iters
        )
        if lat_us < 0:
            raise RuntimeError(f"Native benchmark returned error code: {lat_us}")
        return float(lat_us)

    def get_arena_alloc_count(self) -> int:
        """Returns total native heap allocations performed by thread-local arena pool."""
        if hasattr(self.lib, "forgetune_get_arena_alloc_count"):
            fn = getattr(self.lib, "forgetune_get_arena_alloc_count")
            fn.restype = ctypes.c_size_t
            fn.argtypes = []
            return int(fn())
        return 0

    def reset_arena_alloc_count(self) -> None:
        """Resets the thread-local arena allocation counter to zero."""
        if hasattr(self.lib, "forgetune_reset_arena_alloc_count"):
            fn = getattr(self.lib, "forgetune_reset_arena_alloc_count")
            fn.restype = None
            fn.argtypes = []
            fn()

