# ForgeTune Audit C: Clean Rebuild, Full Automated Tests, and End-to-End Reproducibility Report

**Date:** 2026-10-13  
**Status:** PASS (Clean Rebuild & Full Execution Reproducibility Verified)  
**Auditor:** Independent Systems & Compiler Reviewer  

---

## 1. Environment Baseline Specification

- **Host Operating System:** Windows 11 Enterprise/Pro x64 (`Windows-11-10.0.26200-SP0`, AMD64)
- **Python Runtime:** Python 3.12.14 (`cpython-3.12-windows-x86_64-none`, 64-bit WindowsPE)
- **C++ Compiler:** Zig C++ 0.16.0 (`.venv\Lib\site-packages\ziglang\zig.exe c++` targeting `x86_64-windows-gnu`, C++17)
- **Build System:** CMake 4.4.4 installed in `.venv\Scripts\cmake.exe`; `CMakeLists.txt` and `CMakePresets.json`
- **PyTorch:** PyTorch 2.14.1+cpu
- **pybind11:** Version 3.1.0 installed in `.venv\Lib\site-packages\pybind11\include`
- **NumPy:** Version 2.5.2
- **PyTest:** Version 9.1.1
- **Repository Cleanliness:** Git 2.55.0 clean working tree

---

## 2. Clean Rebuild Protocol Execution

The clean rebuild protocol was executed via `scripts/clean_build.py`:
```bash
python scripts/clean_build.py
```

### 2.1 Rebuild Sequence
1. **Purge Build Artifacts:**
   - Deleted `build/` directory completely.
   - Deleted `bin/` directory completely.
   - Deleted compiled `python/forgetune/_forgetune_cpp.pyd` binary.
2. **Compile Native C++ Test Suite:**
   - Command: `zig c++ -target x86_64-windows-gnu -o bin/forgetune_test_suite.exe src/*.cpp tests/cpp/test_main.cpp -I include -std=c++17 -O2 -w`
   - Exit Code: `0` (Duration: `0.36s`).
3. **Run Native C++ Test Suite:**
   - Command: `bin/forgetune_test_suite.exe`
   - Exit Code: `0` (Duration: `0.12s`).
   - Results: 8 / 8 tests passed.
4. **Compile C++ Compiler Engine CLI:**
   - Command: `zig c++ -target x86_64-windows-gnu -o bin/forgetune_engine.exe src/*.cpp src/main.cpp -I include -std=c++17 -O2 -w`
   - Exit Code: `0` (Duration: `0.33s`).
5. **Compile pybind11 Extension Module:**
   - Command: `zig c++ -target x86_64-windows-gnu -shared -o python/forgetune/_forgetune_cpp.pyd src/bindings.cpp src/ir.cpp src/graph_verifier.cpp src/interpreter.cpp src/pass_manager.cpp src/optimizer.cpp src/fusion.cpp src/codegen.cpp src/json_io.cpp -I include -I <Python_Include> -I <pybind11_Include> <Python_Lib> -std=c++17 -O2 -w`
   - Exit Code: `0` (Duration: `0.47s`).
6. **Execute Python PyTest Suite:**
   - Command: `pytest -v tests/`
   - Exit Code: `0` (Duration: `108.12s`).
   - Results: 44 / 44 tests passed.
7. **Execute TinyMLP Target Workload:**
   - Command: `python examples/tiny_mlp.py`
   - Exit Code: `0` (Duration: `6.12s`).
   - Numerical max absolute error: $2.9802 \times 10^{-8}$.
   - Selected candidate: `fused_ikj_o3` (variant: fused_ikj, fusion: True, compiler: -O3, tile sizes: N/A).

---

## 3. Automated Test Inventory and Results

### 3.1 C++ Native Tests (`bin/forgetune_test_suite.exe`)
1. `IR construction and verification`: **PASSED**
2. `Negative graph verifier diagnostics`: **PASSED**
3. `Reference interpreter numerical evaluation`: **PASSED**
4. `Constant folding pass`: **PASSED**
5. `Dead code elimination pass`: **PASSED**
6. `Operator fusion pass`: **PASSED**
7. `JSON serialization/deserialization round-trip`: **PASSED**
8. `Algebraic simplification and fusion safety`: **PASSED**
*Subtotal: 8 / 8 PASSED.*

### 3.2 Python Test Suite (`pytest -v tests/`)
1. `tests/correctness/test_all_variants.py::test_all_five_kernel_variants[17-31-23]`: **PASSED**
2. `tests/correctness/test_all_variants.py::test_all_five_kernel_variants[16-16-16]`: **PASSED**
3. `tests/correctness/test_all_variants.py::test_all_five_kernel_variants[3-5-7]`: **PASSED**
4. `tests/correctness/test_all_variants.py::test_all_five_kernel_variants[1-48-24]`: **PASSED**
5. `tests/correctness/test_extreme_inputs_all_variants`: **PASSED**
6. `tests/correctness/test_differential.py::test_full_differential_hierarchy`: **PASSED**
7. `tests/correctness/test_edge_cases.py::test_prime_dimensions_and_partial_tiles`: **PASSED**
8. `tests/correctness/test_edge_cases.py::test_batch_size_one`: **PASSED**
9. `tests/correctness/test_edge_cases.py::test_all_zeros_and_all_negatives`: **PASSED**
10. `tests/integration/test_e2e_tinymlp.py::test_tinymlp_full_lifecycle`: **PASSED**
11. `tests/test_autotuner.py::test_autotuner_budget_and_candidates`: **PASSED**
12. `tests/test_autotuner.py::test_autotuning_report_generation`: **PASSED**
13. `tests/unit/test_codegen_and_runtime.py::test_runtime_successful_execution`: **PASSED**
14. `tests/unit/test_codegen_and_runtime.py::test_runtime_invalid_shape_rejection`: **PASSED**
15. `tests/unit/test_codegen_and_runtime.py::test_runtime_null_pointer_contract`: **PASSED**
16. `tests/unit/test_codegen_and_runtime.py::test_runtime_size_mismatch_contract`: **PASSED**
17. `tests/unit/test_codegen_and_runtime.py::test_native_execution_both_bias_forms`: **PASSED**
18. `tests/unit/test_float_serialization.py::test_bit_exact_float32_serialization`: **PASSED**
19. `tests/unit/test_ir.py::test_duplicate_node_id_rejected`: **PASSED**
20. `tests/unit/test_ir.py::test_cyclic_graph_rejected`: **PASSED**
21. `tests/unit/test_ir.py::test_topological_sort_order`: **PASSED**
22. `tests/unit/test_ir.py::test_json_serialization_and_deserialization`: **PASSED**
23. `tests/unit/test_operators.py::test_matmul_valid_dimensions`: **PASSED**
24. `tests/unit/test_operators.py::test_matmul_invalid_dimensions`: **PASSED**
25. `tests/unit/test_operators.py::test_matmul_non_square`: **PASSED**
26. `tests/unit/test_operators.py::test_add_exact_shape`: **PASSED**
27. `tests/unit/test_operators.py::test_add_rank1_vector_broadcasting`: **PASSED**
28. `tests/unit/test_operators.py::test_add_rank2_row_broadcasting`: **PASSED**
29. `tests/unit/test_operators.py::test_add_unsupported_shapes_rejected`: **PASSED**
30. `tests/unit/test_operators.py::test_relu_semantics`: **PASSED**
31. `tests/unit/test_operators.py::test_const_node_storage`: **PASSED**
32. `tests/unit/test_optimizations.py::test_engine_optimization_pipeline`: **PASSED**
33. `tests/unit/test_optimizations.py::test_engine_operator_fusion`: **PASSED**
34. `tests/unit/test_algebraic_simplification_add_zero`: **PASSED**
35. `tests/unit/test_algebraic_simplification_relu_idempotent`: **PASSED**
36. `tests/unit/test_fusion_multi_consumer_rejected`: **PASSED**
37. `tests/unit/test_fusion_output_node_rejected`: **PASSED**
38. `tests/unit/test_pybind11_bindings.py::test_pybind11_ir_and_verifier`: **PASSED**
39. `tests/unit/test_pybind11_bindings.py::test_pybind11_interpreter_against_pytorch`: **PASSED**
40. `tests/unit/test_pybind11_bindings.py::test_pybind11_json_roundtrip`: **PASSED**
41. `tests/unit/test_torch_export_frontend.py::test_torch_export_extraction_and_transposition`: **PASSED**
42. `tests/unit/test_torch_export_frontend.py::test_torch_export_rejects_unsupported_ops`: **PASSED**
43. `tests/unit/test_torch_export_frontend.py::test_torch_export_rejects_rank_3_inputs`: **PASSED**
44. `tests/unit/test_torch_export_frontend.py::test_canonical_tinymlp_exported_aten_targets_and_decomposition`: **PASSED**
*Subtotal: 44 / 44 PASSED.*

**Total Automated Tests:** **52 / 52 PASSED (100% Pass Rate).**

---

## 4. Benchmark Reproducibility

Benchmark reproduction script `scripts/run_benchmarks.py` executed:
- **PyTorch Reference Inference (CPU):** Median Latency: $26.70\ \mu\text{s}$ ($\text{IQR} = 0.60\ \mu\text{s}$).
- **ForgeTune Baseline Naive C++ (`-O2`):** Median Latency: $7.20\ \mu\text{s}$ ($\text{IQR} = 0.30\ \mu\text{s}$).
- **ForgeTune Fused IKJ C++ (`-O2`):** Median Latency: $7.10\ \mu\text{s}$ ($\text{IQR} = 0.20\ \mu\text{s}$).
- **ForgeTune Fused IKJ C++ (`-O3`):** Median Latency: $7.10\ \mu\text{s}$ ($\text{IQR} = 0.20\ \mu\text{s}$).
- **ForgeTune Autotuned Winner C++:** Median Latency: $7.10\ \mu\text{s}$ ($\text{IQR} = 0.20\ \mu\text{s}$).
- **End-to-End Speedup vs PyTorch CPU:** $3.76\times$.
- **In-Kernel Pure C++ Latency (10,000 runs via `std::chrono::steady_clock`):**
  - Baseline Naive Kernel (`naive_ijk`, -O2): $20.1\text{ ns}$ ($0.0201\ \mu\text{s}$)
  - Fused IKJ Kernel (`fused_ikj`, -O2): $17.6\text{ ns}$ ($0.0176\ \mu\text{s}$) $\implies$ **$1.14\times$ Algorithmic Fusion Speedup (-O2 vs -O2)**
  - Fused IKJ Kernel (`fused_ikj`, -O3): $17.6\text{ ns}$ ($0.0176\ \mu\text{s}$) $\implies$ **$1.14\times$ Speedup** (identical to -O2)
  - Autotuned Winner Kernel: $17.6\text{ ns}$ ($0.0176\ \mu\text{s}$)
- **Compiler Safety:** Inline compiler memory barrier (`__asm__ __volatile__("" : "+m"(*(float*)output) : : "memory");`) and outside-timed checksum guarantee execution cannot be optimized away.

All metrics persisted in `reports/benchmark_report.json`.

---

## 5. Audit C Conclusion

Audit C confirms that the ForgeTune repository cleanly rebuilds from scratch, deterministically executes all 52 automated tests with zero failures, and reproduces the end-to-end TinyMLP compilation and autotuning workflow.
