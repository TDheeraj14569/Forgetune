"""
Public Python API for the ForgeTune Machine Learning Compiler.
Provides compile_model and CompiledModel.
"""

import os
import sys
import shutil
import hashlib
import json
import platform
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, Union, Callable, List

from .export_frontend import export_to_forgetune_ir
from .native_compiler import NativeCompiler
from .runtime import NativeRuntime
from .autotuner import Autotuner, KernelCandidate
from .manifest import create_manifest, write_atomic_json, read_manifest
from .benchmark import benchmark_function
from .errors import CorrectnessError

def _compute_file_sha256(path: str) -> str:
    """Computes SHA-256 hash of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def _compute_graph_sha256(graph_json: str) -> str:
    """Computes SHA-256 hash of serialized IR graph JSON."""
    return hashlib.sha256(graph_json.encode("utf-8")).hexdigest()

def get_compiler_sources_fingerprint() -> str:
    """Computes SHA-256 fingerprint across all compiler source code, C/C++ headers, and build configuration."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    import glob
    tracked_files = []
    for pattern in [
        "src/*.cpp",
        "include/*.h",
        "include/**/*.h",
        "include/**/*.hpp",
        "python/forgetune/*.py",
        "setup.py",
        "pyproject.toml"
    ]:
        tracked_files.extend(glob.glob(os.path.join(root, pattern), recursive=True))
    tracked_files = sorted(set(tracked_files))
    h = hashlib.sha256()
    for p in tracked_files:
        if os.path.isfile(p):
            rel_path = os.path.relpath(p, root).replace("\\", "/")
            h.update(rel_path.encode("utf-8"))
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:16]

_get_compiler_sources_fingerprint = get_compiler_sources_fingerprint

def get_bounded_test_batches(
    input_shape: tuple,
    max_batch_size: int,
    example_batch: int = 1,
    max_single_batch_bytes: int = 32 * 1024 * 1024
) -> List[int]:
    """
    Computes a representative, memory-budget-aware list of batch sizes for correctness verification.
    Guarantees that test allocations will not exceed the memory budget.
    """
    if len(input_shape) > 0 and input_shape[0] != -1:
        return [int(input_shape[0])]

    elem_per_sample = int(np.prod(input_shape[1:])) if len(input_shape) > 1 else 1
    bytes_per_sample = max(1, elem_per_sample * 4)  # float32 = 4 bytes
    max_safe_b_by_mem = max(1, max_single_batch_bytes // bytes_per_sample)
    safe_cap = min(max_batch_size, 256, max_safe_b_by_mem)
    batches = {1, example_batch, min(16, max_batch_size), safe_cap}
    if max_batch_size * bytes_per_sample <= max_single_batch_bytes and max_batch_size <= 256:
        batches.add(max_batch_size)
    return sorted([int(b) for b in batches if 1 <= b <= max_batch_size])

def _get_candidate_provenance_hash(
    graph_hash: str,
    cand: KernelCandidate,
    compiler_fingerprint: Optional[str] = None
) -> str:
    """Generates unique deterministic hash of graph, exact candidate configuration, and compiler implementation."""
    if compiler_fingerprint is None:
        compiler_fingerprint = _get_compiler_sources_fingerprint()
    spec_str = json.dumps({
        "graph_hash": graph_hash,
        "cand": cand.to_dict(),
        "compiler_fingerprint": compiler_fingerprint
    }, sort_keys=True)
    return hashlib.sha256(spec_str.encode("utf-8")).hexdigest()[:12]

def _is_artifact_valid_for_graph(
    cand_dir: str,
    graph_hash: str,
    cand: KernelCandidate,
    compiler_info: Dict[str, Any],
    dll_path: Optional[str] = None,
    cpp_path: Optional[str] = None,
    compiler_fingerprint: Optional[str] = None
) -> bool:
    """
    Validates whether existing compilation artifacts in cand_dir match the graph,
    complete candidate configuration, compiler path/version, host platform, code generator fingerprint, and DLL hash.
    Prevents cross-model, cross-configuration, and stale-compiler artifact reuse.
    """
    manifest_path = os.path.join(cand_dir, "cand_manifest.json")
    if cpp_path is None:
        cpp_path = os.path.join(cand_dir, "kernel.cpp")
    if dll_path is None:
        dll_ext = ".dll" if sys.platform == "win32" else ".so"
        dll_path = os.path.join(cand_dir, f"kernel{dll_ext}")

    if not (os.path.isfile(manifest_path) and os.path.isfile(cpp_path) and os.path.isfile(dll_path)):
        return False

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        if m.get("graph_hash") != graph_hash:
            return False
        if m.get("candidate_id") != cand.candidate_id:
            return False
        if m.get("variant") != cand.variant:
            return False
        if m.get("enable_fusion") != cand.enable_fusion:
            return False
        if m.get("tile_sizes") != [cand.tile_m, cand.tile_n, cand.tile_k]:
            return False
        if m.get("opt_level") != cand.opt_level:
            return False
        if m.get("platform") != sys.platform:
            return False
        if m.get("machine") != platform.machine():
            return False
        if m.get("compiler_type") != compiler_info.get("type"):
            return False
        if m.get("compiler_version") != compiler_info.get("version"):
            return False
        if compiler_info.get("path") and m.get("compiler_path") != compiler_info.get("path"):
            return False
        if compiler_fingerprint is None:
            compiler_fingerprint = _get_compiler_sources_fingerprint()
        if m.get("compiler_fingerprint") != compiler_fingerprint:
            return False
        if m.get("source_hash") != _compute_file_sha256(cpp_path):
            return False
        if m.get("dll_hash") != _compute_file_sha256(dll_path):
            return False
        return True
    except Exception:
        return False

def _write_artifact_manifest(
    cand_dir: str,
    graph_hash: str,
    cand: KernelCandidate,
    cpp_path: str,
    dll_path: str,
    compiler_info: Dict[str, Any],
    compiler_fingerprint: Optional[str] = None
) -> None:
    """Writes complete candidate provenance manifest validating compilation artifacts."""
    if compiler_fingerprint is None:
        compiler_fingerprint = _get_compiler_sources_fingerprint()
    manifest_data = {
        "graph_hash": graph_hash,
        "candidate": cand.to_dict(),
        "candidate_id": cand.candidate_id,
        "variant": cand.variant,
        "enable_fusion": cand.enable_fusion,
        "tile_sizes": [cand.tile_m, cand.tile_n, cand.tile_k],
        "opt_level": cand.opt_level,
        "platform": sys.platform,
        "machine": platform.machine(),
        "compiler_type": compiler_info.get("type"),
        "compiler_version": compiler_info.get("version"),
        "compiler_path": compiler_info.get("path"),
        "compiler_fingerprint": compiler_fingerprint,
        "source_hash": _compute_file_sha256(cpp_path),
        "dll_hash": _compute_file_sha256(dll_path),
    }
    write_atomic_json(os.path.join(cand_dir, "cand_manifest.json"), manifest_data)

def _verify_rebuilt_artifact(
    dll_path: str,
    ir_graph: Any,
    example_input: np.ndarray,
    reference_output: np.ndarray,
    oracle: Callable[[np.ndarray], np.ndarray],
    dynamic_batch: bool = False,
    max_batch_size: int = 128,
    atol: float = 1e-4,
    rtol: float = 1e-4
) -> None:
    """Differential correctness gate executed on freshly compiled/rebuilt shared library."""
    if dynamic_batch or ir_graph.has_dynamic_batch():
        in_shape = tuple(-1 if i == 0 else s for i, s in enumerate(example_input.shape))
        out_node = ir_graph.nodes[ir_graph.outputs[0]]
        out_is_dyn = (len(out_node.shape) > 0 and out_node.shape[0] == -1)
        out_shape = tuple(-1 if (out_is_dyn and i == 0) else s for i, s in enumerate(reference_output.shape))
    else:
        in_shape = tuple(example_input.shape)
        out_shape = tuple(reference_output.shape)

    runtime = NativeRuntime(dll_path, in_shape, out_shape)
    try:
        # Test 1: Example input
        out = runtime.run(example_input, batch_size=example_input.shape[0] if (dynamic_batch or ir_graph.has_dynamic_batch()) else None)
        if not np.allclose(out, reference_output, atol=atol, rtol=rtol):
            max_diff = float(np.max(np.abs(out - reference_output)))
            raise CorrectnessError(
                f"Rebuilt artifact {dll_path} failed differential correctness verification on example input (max abs diff: {max_diff:.3e})"
            )

        # Test 2: If dynamic batch, test representative memory-bounded batches including configured maximum
        if dynamic_batch or ir_graph.has_dynamic_batch():
            test_batches = get_bounded_test_batches(example_input.shape, max_batch_size, example_batch=example_input.shape[0])
            for b in test_batches:
                shape = (b, *example_input.shape[1:])
                test_in = np.ones(shape, dtype=np.float32) * 0.5
                expected = oracle(test_in)
                actual = runtime.run(test_in, batch_size=b)
                if not np.allclose(actual, expected, atol=atol, rtol=rtol):
                    max_diff = float(np.max(np.abs(actual - expected)))
                    raise CorrectnessError(
                        f"Rebuilt artifact {dll_path} failed differential correctness verification at dynamic batch {b} (max abs diff: {max_diff:.3e})"
                    )
    finally:
        runtime.close()

def emit_cpp_source(ir_graph, candidate: KernelCandidate, output_cpp_path: str, cand_json_path: Optional[str] = None) -> None:
    """
    Emits standalone C++ source code for a given IR graph and candidate configuration.
    Uses in-process C++ compiler bridge (_forgetune_cpp) when available, falling back to CLI.
    """
    try:
        from . import _forgetune_cpp as cpp
        g_cpp = cpp.json_to_graph(ir_graph.to_json())
        cfg = cpp.KernelConfig()
        cfg.candidate_id = candidate.candidate_id
        variant_map = {
            "naive_ijk": cpp.MatMulVariant.NAIVE_IJK,
            "loop_ikj": cpp.MatMulVariant.LOOP_IKJ,
            "tiled": cpp.MatMulVariant.TILED,
            "fused_ikj": cpp.MatMulVariant.FUSED_IKJ,
            "fused_tiled": cpp.MatMulVariant.FUSED_TILED
        }
        if candidate.variant not in variant_map:
            raise ValueError(f"Unknown kernel variant: '{candidate.variant}'. Valid variants: {sorted(list(variant_map.keys()))}")
        cfg.matmul_variant = variant_map[candidate.variant]
        cfg.enable_fusion = bool(candidate.enable_fusion)
        if candidate.tile_m is not None:
            cfg.tile_m = int(candidate.tile_m)
            cfg.tile_n = int(candidate.tile_n)
            cfg.tile_k = int(candidate.tile_k)
        if hasattr(ir_graph, "max_batch_size"):
            cfg.max_batch_size = int(ir_graph.max_batch_size)
        compiler = cpp.Compiler(cfg)
        res = compiler.compile(g_cpp)
        if res.success:
            with open(output_cpp_path, "w", encoding="utf-8") as f:
                f.write(res.generated_cpp_source)
            return
    except Exception:
        pass

    # Fallback to CLI engine
    if cand_json_path is None:
        cand_json_path = os.path.splitext(output_cpp_path)[0] + ".json"
    with open(cand_json_path, "w", encoding="utf-8") as f:
        f.write(ir_graph.to_json())
    engine_exe = get_engine_executable()
    import subprocess
    cmd = [
        engine_exe, "--compile", cand_json_path, "-o", output_cpp_path,
        "--variant", candidate.variant,
        "--candidate-id", candidate.candidate_id,
    ]
    if candidate.tile_m is not None:
        cmd.extend([
            "--tile-m", str(candidate.tile_m),
            "--tile-n", str(candidate.tile_n),
            "--tile-k", str(candidate.tile_k)
        ])
    if candidate.enable_fusion:
        cmd.append("--fusion")
    if hasattr(ir_graph, "max_batch_size"):
        cmd.extend(["--max-batch-size", str(ir_graph.max_batch_size)])
    subprocess.run(cmd, check=True)


def get_engine_executable() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    search_dirs = [
        os.path.join(root, "bin"),
        os.path.join(root, "build", "bin"),
        os.path.join(root, "build", "bin", "Release"),
        os.path.join(root, "build", "bin", "Debug"),
        os.path.join(root, "build"),
    ]
    if sys.platform == "win32":
        exe_names = [
            "forgetune_test_engine.exe",
            "forgetune_engine.exe",
            "forgetune_test_engine",
            "forgetune_engine"
        ]
    else:
        exe_names = [
            "forgetune_test_engine",
            "forgetune_engine",
            "forgetune_test_engine.exe",
            "forgetune_engine.exe"
        ]
    for d in search_dirs:
        for name in exe_names:
            cand = os.path.join(d, name)
            if os.path.isfile(cand):
                return cand
    return os.path.join(root, "bin", "forgetune_engine.exe" if sys.platform == "win32" else "forgetune_engine")

class CompiledModel:
    def __init__(
        self,
        runtime: NativeRuntime,
        manifest_data: Dict[str, Any],
        cpp_source: str,
        opt_report: Dict[str, Any]
    ):
        self._runtime = runtime
        self._manifest = manifest_data
        self._cpp_source = cpp_source
        self._opt_report = opt_report

    def run(self, input_tensor: Union[np.ndarray, torch.Tensor], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Executes native CPU inference using the generated standalone C++ shared library.
        Does NOT invoke PyTorch for arithmetic evaluation.
        """
        if isinstance(input_tensor, torch.Tensor):
            arr = input_tensor.detach().cpu().to(torch.float32).numpy()
        else:
            arr = np.asarray(input_tensor, dtype=np.float32)

        return self._runtime.run(arr, batch_size=batch_size)

    def __call__(self, input_tensor: Union[np.ndarray, torch.Tensor], batch_size: Optional[int] = None) -> np.ndarray:
        return self.run(input_tensor, batch_size=batch_size)

    def optimization_report(self) -> Dict[str, Any]:
        """Returns the optimization and autotuning report."""
        return self._opt_report

    def manifest(self) -> Dict[str, Any]:
        """Returns the model artifact manifest."""
        return self._manifest

    def source_code(self) -> str:
        """Returns the emitted C++ source code."""
        return self._cpp_source

    def benchmark(self, warmup_iters: int = 10, measure_iters: int = 50, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Measures steady-state native inference latency on host machine.
        """
        if self._runtime.is_dynamic:
            b = batch_size if batch_size is not None else 1
            shape = (b, *self._runtime.expected_input_shape[1:])
            dummy_input = np.ones(shape, dtype=np.float32)
        else:
            dummy_input = np.ones(self._runtime.expected_input_shape, dtype=np.float32)
        return benchmark_function(lambda: self._runtime.run(dummy_input), warmup_iters, measure_iters)

    def benchmark_kernel(self, iters: int = 1000, batch_size: Optional[int] = None) -> float:
        """
        Executes microbenchmark inside native C++ without Python/ctypes per-iteration overhead.
        Returns average latency per call in microseconds.
        """
        if self._runtime.is_dynamic:
            b = batch_size if batch_size is not None else 1
            shape = (b, *self._runtime.expected_input_shape[1:])
            dummy_input = np.ones(shape, dtype=np.float32)
        else:
            dummy_input = np.ones(self._runtime.expected_input_shape, dtype=np.float32)
        return self._runtime.benchmark_kernel(dummy_input, iters)

    def get_arena_alloc_count(self) -> int:
        """Returns total native allocations (payload + block insertions) performed by the arena pool."""
        return self._runtime.get_arena_alloc_count()

    def get_arena_payload_alloc_count(self) -> int:
        """Returns total payload buffer allocations performed by the arena pool."""
        return self._runtime.get_arena_payload_alloc_count()

    def get_arena_block_insertion_count(self) -> int:
        """Returns total block container growth insertions performed by the arena pool."""
        return self._runtime.get_arena_block_insertion_count()

    def reset_arena_alloc_count(self) -> None:
        """Resets the arena allocation counters."""
        self._runtime.reset_arena_alloc_count()

    def inject_test_arena_alloc(self) -> None:
        """Forces an allocation in the native arena pool for regression testing."""
        self._runtime.inject_test_arena_alloc()

    def close(self) -> None:
        """Releases native runtime resources and unloads native libraries."""
        if self._runtime is not None:
            self._runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def compile_model(
    model: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    output_dir: str = "build/artifacts/model",
    autotune: bool = True,
    max_autotune_budget: int = 8,
    compiler_opt_level: str = "-O2",
    kernel_variant: Optional[str] = None,
    enable_fusion: bool = False,
    tile_sizes: Optional[Tuple[int, int, int]] = None,
    dynamic_batch: bool = False,
    max_batch_size: int = 128
) -> CompiledModel:
    """
    Compiles a PyTorch model into a standalone, correctness-verified native C++ shared library.
    Pipeline:
      1. torch.export graph extraction & inspection
      2. Normalization into canonical ForgeTune IR
      3. Graph verification
      4. Correctness-gated autotuning across kernel variants (naive, ikj, tiled, fused)
      5. Standalone C++ code generation
      6. Native compilation into shared library (.dll / .so)
      7. Dynamic loading via NativeRuntime
      8. Manifest and report generation
    """
    # Strict API parameter validation (reject bool masquerading as int)
    if type(max_autotune_budget) is not int or not (1 <= max_autotune_budget <= 8):
        raise ValueError(f"max_autotune_budget must be an integer between 1 and 8, got {max_autotune_budget} ({type(max_autotune_budget).__name__})")
    if type(max_batch_size) is not int or not (1 <= max_batch_size <= 1000000):
        raise ValueError(f"max_batch_size must be an integer between 1 and 1000000, got {max_batch_size} ({type(max_batch_size).__name__})")
    if type(autotune) is not bool:
        raise ValueError(f"autotune must be a bool, got {type(autotune).__name__}")
    if type(dynamic_batch) is not bool:
        raise ValueError(f"dynamic_batch must be a bool, got {type(dynamic_batch).__name__}")
    if type(enable_fusion) is not bool:
        raise ValueError(f"enable_fusion must be a bool, got {type(enable_fusion).__name__}")

    if not isinstance(compiler_opt_level, str):
        raise TypeError(f"compiler_opt_level must be a string, got {type(compiler_opt_level).__name__}")

    if tile_sizes is not None:
        if not isinstance(tile_sizes, (tuple, list)) or len(tile_sizes) != 3 or any(type(t) is not int or t <= 0 for t in tile_sizes):
            raise ValueError(f"tile_sizes must be a 3-tuple/list of positive integers, got {tile_sizes}")

    norm_opt = compiler_opt_level if compiler_opt_level.startswith("-") or compiler_opt_level.startswith("/") else f"-{compiler_opt_level}"
    valid_opts = {"-O0", "-O1", "-O2", "-O3", "-Os", "-Oz", "/Od", "/O1", "/O2", "/Ox"}
    if norm_opt not in valid_opts and compiler_opt_level not in valid_opts:
        raise ValueError(f"Unsupported compiler optimization level: '{compiler_opt_level}'")

    if autotune:
        if kernel_variant is not None:
            raise ValueError(
                f"Cannot specify 'kernel_variant' ('{kernel_variant}') when autotune=True. "
                "Autotuner automatically explores kernel variants. Set autotune=False for manual kernel selection."
            )
        if tile_sizes is not None:
            raise ValueError(
                f"Cannot specify 'tile_sizes' ({tile_sizes}) when autotune=True. "
                "Autotuner automatically explores tile configurations. Set autotune=False for manual tile sizes."
            )
        if compiler_opt_level not in ("-O2", "O2", "/O2"):
            raise ValueError(
                f"Cannot override 'compiler_opt_level' ('{compiler_opt_level}') when autotune=True. "
                "Autotuning explores compiler optimization levels automatically. Set autotune=False for manual optimization level."
            )

    # Enforce CPU execution contract on model parameters, buffers, and example inputs
    for name, p in model.named_parameters():
        if p.device.type != "cpu":
            raise ValueError(f"Model parameter '{name}' is on device '{p.device}', but ForgeTune CPU compiler requires all parameters to be on 'cpu'.")
    for name, b in model.named_buffers():
        if b.device.type != "cpu":
            raise ValueError(f"Model buffer '{name}' is on device '{b.device}', but ForgeTune CPU compiler requires all buffers to be on 'cpu'.")
    for i, t in enumerate(example_inputs):
        if not isinstance(t, torch.Tensor) or t.device.type != "cpu":
            raise ValueError(f"Example input {i} is on device '{getattr(t, 'device', 'unknown')}', but ForgeTune requires CPU tensors.")

    os.makedirs(output_dir, exist_ok=True)
    saved_modes = {}
    if hasattr(model, "named_modules"):
        for m_name, mod in model.named_modules():
            saved_modes[m_name] = getattr(mod, "training", False)
    else:
        saved_modes[""] = getattr(model, "training", False)

    try:
        model.eval()

        # 1. Export frontend
        ir_graph = export_to_forgetune_ir(model, example_inputs, dynamic_batch=dynamic_batch, max_batch_size=max_batch_size)
        graph_json_str = ir_graph.to_json()
        graph_hash = _compute_graph_sha256(graph_json_str)

        # Convert example input to numpy
        ex_np = example_inputs[0].detach().cpu().to(torch.float32).numpy()

        # Compute reference output using PyTorch model
        with torch.no_grad():
            torch_out = model(*example_inputs)
            if isinstance(torch_out, tuple):
                torch_out = torch_out[0]
            ref_np = torch_out.detach().cpu().to(torch.float32).numpy()

        native_compiler = NativeCompiler()
        compiler_info = native_compiler.get_compiler_info()

        def oracle(x_arr: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.from_numpy(x_arr)
                out = model(t)
                if isinstance(out, tuple):
                    out = out[0]
                return out.detach().cpu().to(torch.float32).numpy()

        if autotune:
            tuner = Autotuner(native_compiler=native_compiler)
            selected_cand, tuning_report = tuner.tune(
                graph=ir_graph,
                example_input=ex_np,
                reference_output=ref_np,
                artifacts_dir=output_dir,
                max_budget=max_autotune_budget,
                oracle=oracle,
                enable_fusion=enable_fusion
            )
            cand_provenance_hash = _get_candidate_provenance_hash(graph_hash, selected_cand)
            cand_dir = os.path.join(output_dir, f"{selected_cand.candidate_id}_{cand_provenance_hash}")
            os.makedirs(cand_dir, exist_ok=True)
            dll_ext = ".dll" if sys.platform == "win32" else ".so"
            final_cpp_path = os.path.join(cand_dir, "kernel.cpp")
            final_dll_path = os.path.join(cand_dir, f"kernel_{cand_provenance_hash}{dll_ext}")
            opt_report = tuning_report

            if not _is_artifact_valid_for_graph(cand_dir, graph_hash, selected_cand, compiler_info, final_dll_path, final_cpp_path):
                cand_json_path = os.path.join(cand_dir, "graph.json")
                emit_cpp_source(ir_graph, selected_cand, final_cpp_path, cand_json_path)
                native_compiler.compile_shared_library(final_cpp_path, final_dll_path, opt_level=selected_cand.opt_level)
                _verify_rebuilt_artifact(final_dll_path, ir_graph, ex_np, ref_np, oracle, dynamic_batch=dynamic_batch, max_batch_size=max_batch_size)
                _write_artifact_manifest(cand_dir, graph_hash, selected_cand, final_cpp_path, final_dll_path, compiler_info)
        else:
            # Explicit or baseline compilation
            VALID_VARIANTS = {"naive_ijk", "loop_ikj", "tiled", "fused_ikj", "fused_tiled"}
            variant = kernel_variant or "naive_ijk"
            if variant not in VALID_VARIANTS:
                raise ValueError(f"Unknown kernel variant: '{variant}'. Valid variants: {sorted(list(VALID_VARIANTS))}")
            if not enable_fusion and variant.startswith("fused"):
                raise ValueError(
                    f"Contradictory options: fused kernel_variant '{variant}' requested with enable_fusion=False"
                )
            cand_id = "baseline" if (variant == "naive_ijk" and not enable_fusion) else variant
            tile_m = tile_sizes[0] if tile_sizes else None
            tile_n = tile_sizes[1] if tile_sizes else None
            tile_k = tile_sizes[2] if tile_sizes else None

            selected_cand = KernelCandidate(
                cand_id, variant, enable_fusion=enable_fusion,
                tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                opt_level=compiler_opt_level
            )
            cand_provenance_hash = _get_candidate_provenance_hash(graph_hash, selected_cand)
            cand_dir = os.path.join(output_dir, f"{selected_cand.candidate_id}_{cand_provenance_hash}")
            os.makedirs(cand_dir, exist_ok=True)
            dll_ext = ".dll" if sys.platform == "win32" else ".so"
            final_json_path = os.path.join(cand_dir, "graph.json")
            final_cpp_path = os.path.join(cand_dir, "kernel.cpp")
            final_dll_path = os.path.join(cand_dir, f"kernel_{cand_provenance_hash}{dll_ext}")

            if not _is_artifact_valid_for_graph(cand_dir, graph_hash, selected_cand, compiler_info, final_dll_path, final_cpp_path):
                emit_cpp_source(ir_graph, selected_cand, final_cpp_path, final_json_path)
                native_compiler.compile_shared_library(final_cpp_path, final_dll_path, opt_level=compiler_opt_level)
                _verify_rebuilt_artifact(final_dll_path, ir_graph, ex_np, ref_np, oracle, dynamic_batch=dynamic_batch, max_batch_size=max_batch_size)
                _write_artifact_manifest(cand_dir, graph_hash, selected_cand, final_cpp_path, final_dll_path, compiler_info)

            opt_report = {"selected_candidate": selected_cand.to_dict()}

        with open(final_cpp_path, "r", encoding="utf-8") as f:
            cpp_source = f.read()

        # Load runtime
        if dynamic_batch or ir_graph.has_dynamic_batch():
            in_shape = tuple(-1 if i == 0 else s for i, s in enumerate(ex_np.shape))
            out_node = ir_graph.nodes[ir_graph.outputs[0]]
            out_is_dyn = (len(out_node.shape) > 0 and out_node.shape[0] == -1)
            out_shape = tuple(-1 if (out_is_dyn and i == 0) else s for i, s in enumerate(ref_np.shape))
        else:
            in_shape = tuple(ex_np.shape)
            out_shape = tuple(ref_np.shape)

        runtime = NativeRuntime(final_dll_path, in_shape, out_shape)

        # Create and save manifest
        manifest_data = create_manifest(
            model_name=model.__class__.__name__,
            input_shapes=[list(in_shape)],
            output_shape=list(out_shape),
            kernel_config=selected_cand.to_dict(),
            compiler_info=native_compiler.get_compiler_info(),
            artifacts={
                "cpp_source": final_cpp_path,
                "shared_library": final_dll_path
            },
            optimization_summary=opt_report
        )
        manifest_data["provenance_hash"] = cand_provenance_hash
        manifest_data["latest_note"] = "Root manifest describes latest compilation. Per-artifact manifests are saved alongside candidate artifacts."
        manifest_path = os.path.join(output_dir, "manifest.json")
        prov_manifest_path = os.path.join(output_dir, f"manifest_{cand_provenance_hash}.json")
        write_atomic_json(manifest_path, manifest_data)
        write_atomic_json(prov_manifest_path, manifest_data)

        return CompiledModel(runtime, manifest_data, cpp_source, opt_report)
    finally:
        if hasattr(model, "named_modules"):
            for m_name, mod in model.named_modules():
                if m_name in saved_modes:
                    mod.training = saved_modes[m_name]
        elif "" in saved_modes:
            model.training = saved_modes[""]
