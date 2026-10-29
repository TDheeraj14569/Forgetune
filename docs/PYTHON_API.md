# ForgeTune v2.0 Python API Reference

This document outlines the public Python interface of ForgeTune v2.0.

---

## 1. Top-Level Function: `compile_model`

```python
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
```

### Parameters:
- `model` (`torch.nn.Module`): Eager PyTorch model instance to compile.
- `example_inputs` (`Tuple[torch.Tensor, ...]`): Representative sample input tensors.
- `output_dir` (`str`): Directory where generated C++ source, shared library, and manifest are saved.
- `autotune` (`bool`): If `True`, runs correctness-gated autotuning across candidate configurations.
- `max_autotune_budget` (`int`): Maximum number of candidate kernel configurations to evaluate.
- `compiler_opt_level` (`str`): Native C++ optimization flags (e.g., `"-O2"`, `"-O3"`).
- `kernel_variant` (`Optional[str]`): Manually specify variant (`"naive_ijk"`, `"loop_ikj"`, `"tiled"`, `"fused_ikj"`, `"fused_tiled"`).
- `enable_fusion` (`bool`): Whether to enable operator fusion passes.
- `tile_sizes` (`Optional[Tuple[int, int, int]]`): Tile sizes `(tile_m, tile_n, tile_k)` for tiled variants.
- `dynamic_batch` (`bool`): When `True`, generates dynamic batch kernel supporting $B \in [1, max\_batch\_size]$.
- `max_batch_size` (`int`): Upper bound for dynamic batch dimension (default `128`).

### Returns:
- `CompiledModel`: Executable wrapper holding native runtime handle, manifest, and C++ source.

---

## 2. Class: `CompiledModel`

Wrapper object returned by `compile_model`.

### Methods:

#### `run(input_tensor: Union[np.ndarray, torch.Tensor], batch_size: Optional[int] = None) -> np.ndarray`
Executes native CPU inference using the generated standalone C++ shared library without calling PyTorch arithmetic.
- `input_tensor`: Input data matching expected feature dimensions. In dynamic models, leading batch size $B$ can be any integer in $[1, max\_batch\_size]$ (default `128`).
- Returns: `np.ndarray` contiguous float32 tensor of shape `(B, *output_features)`.

#### `benchmark(warmup_iters: int = 10, measure_iters: int = 50, batch_size: Optional[int] = None) -> Dict[str, Any]`
Measures end-to-end Python inference latency (including ctypes dispatch) using high-resolution monotonic clocks. Returns a dict containing:
- `median_latency_us`, `mean_latency_us`, `min_latency_us`, `max_latency_us`, `std_dev_us`, `iqr_us`.

#### `benchmark_kernel(iters: int = 1000, batch_size: Optional[int] = None) -> float`
Executes microbenchmark running entirely inside native C++ via `std::chrono::steady_clock` with zero ctypes or Python interpreter per-call overhead. Returns average kernel latency in microseconds.

#### `optimization_report() -> Dict[str, Any]`
Returns the autotuning and optimization summary report.

#### `manifest() -> Dict[str, Any]`
Returns the model artifact manifest dictionary.

#### `source_code() -> str`
Returns the emitted standalone C++17 source code string.

---

## 3. Frontend Function: `export_to_forgetune_ir`

```python
def export_to_forgetune_ir(
    model: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    dynamic_batch: bool = False,
    max_batch_size: int = 128
) -> Graph:
```
Exports a PyTorch model via ahead-of-time `torch.export.export` and converts it into canonical ForgeTune IR.

---

## 4. Class: `NativeRuntime`

Low-level ctypes loader for compiled ForgeTune `.dll` binaries:
```python
runtime = NativeRuntime(dll_path, expected_input_shape, expected_output_shape)
output = runtime.run(input_array)
```
- Direct artifact loading with full security metadata preservation (no stream deletion, no temporary staging). Bounded retries strictly for transient file contention.
- Supports versioned C ABI dispatch: `forgetune_run_v2` (explicit 64-bit dynamic batch ABI), `forgetune_run_v1` (compatibility entry point for static or inferred-batch execution), and legacy `forgetune_run`.
- Validates shapes, bounds, dtypes, and null pointers before invocation.
