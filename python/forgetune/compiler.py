"""
Public Python API for the ForgeTune Machine Learning Compiler.
Provides compile_model and CompiledModel.
"""

import os
import shutil
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, Union

from .export_frontend import export_to_forgetune_ir
from .native_compiler import NativeCompiler
from .runtime import NativeRuntime
from .autotuner import Autotuner, KernelCandidate
from .manifest import create_manifest, write_atomic_json, read_manifest
from .benchmark import benchmark_function

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
        cfg.matmul_variant = variant_map.get(candidate.variant, cpp.MatMulVariant.NAIVE_IJK)
        cfg.enable_fusion = bool(candidate.enable_fusion)
        if candidate.tile_m is not None:
            cfg.tile_m = int(candidate.tile_m)
            cfg.tile_n = int(candidate.tile_n)
            cfg.tile_k = int(candidate.tile_k)
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
    subprocess.run(cmd, check=True)


def get_engine_executable() -> str:
    bin_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin"))
    for name in ("forgetune_test_engine.exe", "forgetune_engine.exe"):
        cand = os.path.join(bin_dir, name)
        if os.path.isfile(cand):
            return cand
    return os.path.join(bin_dir, "forgetune_engine.exe")

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
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # 1. Export frontend
    ir_graph = export_to_forgetune_ir(model, example_inputs, dynamic_batch=dynamic_batch, max_batch_size=max_batch_size)

    # Convert example input to numpy
    ex_np = example_inputs[0].detach().cpu().to(torch.float32).numpy()

    # Compute reference output using PyTorch model
    with torch.no_grad():
        torch_out = model(*example_inputs)
        if isinstance(torch_out, tuple):
            torch_out = torch_out[0]
        ref_np = torch_out.detach().cpu().to(torch.float32).numpy()

    native_compiler = NativeCompiler()

    if autotune:
        tuner = Autotuner(native_compiler=native_compiler)
        selected_cand, tuning_report = tuner.tune(
            graph=ir_graph,
            example_input=ex_np,
            reference_output=ref_np,
            artifacts_dir=output_dir,
            max_budget=max_autotune_budget
        )
        cand_id = selected_cand.candidate_id
        final_cpp_path = os.path.join(output_dir, cand_id, "kernel.cpp")
        final_dll_path = os.path.join(output_dir, cand_id, "kernel.dll")
        opt_report = tuning_report

        if not os.path.isfile(final_cpp_path) or not os.path.isfile(final_dll_path):
            cand_dir = os.path.join(output_dir, cand_id)
            os.makedirs(cand_dir, exist_ok=True)
            cand_json_path = os.path.join(cand_dir, "graph.json")
            emit_cpp_source(ir_graph, selected_cand, final_cpp_path, cand_json_path)
            native_compiler.compile_shared_library(final_cpp_path, final_dll_path, opt_level=selected_cand.opt_level)
    else:
        # Explicit or baseline compilation
        variant = kernel_variant or "naive_ijk"
        cand_id = "baseline" if (variant == "naive_ijk" and not enable_fusion) else variant
        tile_m = tile_sizes[0] if tile_sizes else None
        tile_n = tile_sizes[1] if tile_sizes else None
        tile_k = tile_sizes[2] if tile_sizes else None

        selected_cand = KernelCandidate(
            cand_id, variant, enable_fusion=enable_fusion,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            opt_level=compiler_opt_level
        )
        cand_dir = os.path.join(output_dir, cand_id)
        os.makedirs(cand_dir, exist_ok=True)
        final_json_path = os.path.join(cand_dir, "graph.json")
        final_cpp_path = os.path.join(cand_dir, "kernel.cpp")
        final_dll_path = os.path.join(cand_dir, "kernel.dll")

        emit_cpp_source(ir_graph, selected_cand, final_cpp_path, final_json_path)
        native_compiler.compile_shared_library(final_cpp_path, final_dll_path, opt_level=compiler_opt_level)
        opt_report = {"selected_candidate": selected_cand.to_dict()}

    with open(final_cpp_path, "r", encoding="utf-8") as f:
        cpp_source = f.read()

    # Load runtime
    if dynamic_batch or ir_graph.has_dynamic_batch():
        in_shape = tuple(-1 if i == 0 else s for i, s in enumerate(ex_np.shape))
        out_shape = tuple(-1 if i == 0 else s for i, s in enumerate(ref_np.shape))
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
    manifest_path = os.path.join(output_dir, "manifest.json")
    write_atomic_json(manifest_path, manifest_data)

    return CompiledModel(runtime, manifest_data, cpp_source, opt_report)
