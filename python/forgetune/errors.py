"""
ForgeTune exception hierarchy and error classes.
"""

import builtins

class ForgeTuneError(Exception):
    """Base exception for all ForgeTune errors."""
    pass

class UnsupportedOpError(ForgeTuneError):
    """Raised when an unsupported operator is encountered in the frontend."""
    def __init__(self, op_name: str, node_info: str = ""):
        msg = f"Unsupported operator: {op_name}"
        if node_info:
            msg += f" (node details: {node_info})"
        super().__init__(msg)
        self.op_name = op_name
        self.node_info = node_info

class InvalidGraphError(ForgeTuneError):
    """Raised when an IR graph fails verification."""
    pass

class InvalidShapeError(ForgeTuneError):
    """Raised when tensor shapes are incompatible with operation contracts."""
    pass

class CodeGenError(ForgeTuneError):
    """Raised when C++ code generation fails."""
    pass

class CompilationError(ForgeTuneError):
    """Raised when native C++ compilation fails."""
    def __init__(self, message: str, stdout: str = "", stderr: str = "", command: list = None):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.command = command or []

class CorrectnessError(ForgeTuneError):
    """Raised when an implementation fails the numerical correctness gate."""
    def __init__(self, message: str, max_abs_diff: float = 0.0, max_rel_diff: float = 0.0):
        super().__init__(message)
        self.max_abs_diff = max_abs_diff
        self.max_rel_diff = max_rel_diff

class AutotuningError(ForgeTuneError):
    """Raised when autotuning fails to find any valid candidate."""
    pass

class ForgeTuneRuntimeError(ForgeTuneError, builtins.RuntimeError):
    """Raised when native runtime execution fails."""
    pass

# Alias for backwards compatibility
RuntimeError = ForgeTuneRuntimeError
