"""
ForgeTune: Specification-Driven, Correctness-Checked, Self-Optimizing Machine-Learning Compiler.
"""

from .compiler import compile_model, CompiledModel
from .graph_adapter import Graph, Node
from .export_frontend import export_to_forgetune_ir
from .native_compiler import NativeCompiler
from .runtime import NativeRuntime
from .autotuner import Autotuner, KernelCandidate
from .benchmark import benchmark_function, get_machine_metadata
from .manifest import create_manifest, read_manifest, write_atomic_json
from .errors import (
    ForgeTuneError,
    UnsupportedOpError,
    InvalidGraphError,
    InvalidShapeError,
    CodeGenError,
    CompilationError,
    CorrectnessError,
    AutotuningError,
    RuntimeError
)

__version__ = "2.0.0"
__all__ = [
    "compile_model",
    "CompiledModel",
    "Graph",
    "Node",
    "export_to_forgetune_ir",
    "NativeCompiler",
    "NativeRuntime",
    "Autotuner",
    "KernelCandidate",
    "benchmark_function",
    "get_machine_metadata",
    "create_manifest",
    "read_manifest",
    "write_atomic_json",
    "ForgeTuneError",
    "UnsupportedOpError",
    "InvalidGraphError",
    "InvalidShapeError",
    "CodeGenError",
    "CompilationError",
    "CorrectnessError",
    "AutotuningError",
    "RuntimeError",
]
