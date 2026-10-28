"""
Correctness-gated autotuning engine for ForgeTune.
Generates candidate kernel configurations, compiles them, validates them against
independent reference outputs via strict numerical correctness gates, benchmarks
passing candidates on the host machine, and selects the optimal plan with caching.
"""

import os
import hashlib
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .graph_adapter import Graph
from .native_compiler import NativeCompiler
from .runtime import NativeRuntime
from .benchmark import benchmark_function, get_machine_metadata
from .manifest import write_atomic_json
from .errors import CorrectnessError, AutotuningError

class KernelCandidate:
    def __init__(
        self,
        candidate_id: str,
        variant: str,
        enable_fusion: bool = False,
        tile_m: Optional[int] = None,
        tile_n: Optional[int] = None,
        tile_k: Optional[int] = None,
        opt_level: str = "-O2",
        tile_sizes: Optional[Any] = None
    ):
        self.candidate_id = candidate_id
        self.variant = variant
        self.enable_fusion = enable_fusion
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.opt_level = opt_level
        if tile_sizes is not None and self.tile_m is None and isinstance(tile_sizes, (list, tuple)):
            self.tile_m, self.tile_n, self.tile_k = tile_sizes

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "candidate_id": self.candidate_id,
            "variant": self.variant,
            "enable_fusion": self.enable_fusion,
            "opt_level": self.opt_level
        }
        if self.variant in ("tiled", "fused_tiled") and self.tile_m is not None:
            d["tile_m"] = self.tile_m
            d["tile_n"] = self.tile_n
            d["tile_k"] = self.tile_k
            d["tile_sizes"] = (self.tile_m, self.tile_n, self.tile_k)
        else:
            d["tile_sizes"] = None
        return d

class Autotuner:
    def __init__(
        self,
        native_compiler: Optional[NativeCompiler] = None,
        cache_dir: Optional[str] = None,
        atol: float = 1e-4,
        rtol: float = 1e-4
    ):
        self.compiler = native_compiler or NativeCompiler()
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "build", "tuning_cache")
        self.atol = atol
        self.rtol = rtol

    def get_candidate_space(self, max_budget: int = 8) -> List[KernelCandidate]:
        """
        Defines a bounded, deterministic search space of kernel configurations.
        """
        if not isinstance(max_budget, int) or max_budget <= 0:
            raise ValueError(f"max_budget must be a positive integer, got {max_budget}")
        candidates = [
            # 1. Baseline Naive
            KernelCandidate("baseline_naive", "naive_ijk", enable_fusion=False, opt_level="-O2"),
            # 2. Alternate loop ordering (IKJ)
            KernelCandidate("loop_ikj", "loop_ikj", enable_fusion=False, opt_level="-O2"),
            # 3. Tiled execution (tile 8)
            KernelCandidate("tiled_8", "tiled", enable_fusion=False, tile_m=8, tile_n=8, tile_k=8, opt_level="-O2"),
            # 4. Tiled execution (tile 16)
            KernelCandidate("tiled_16", "tiled", enable_fusion=False, tile_m=16, tile_n=16, tile_k=16, opt_level="-O2"),
            # 5. Fused kernel (IKJ)
            KernelCandidate("fused_ikj", "fused_ikj", enable_fusion=True, opt_level="-O2"),
            # 6. Fused kernel (Tiled 8)
            KernelCandidate("fused_tiled_8", "fused_tiled", enable_fusion=True, tile_m=8, tile_n=8, tile_k=8, opt_level="-O2"),
            # 7. Fused kernel (Tiled 16)
            KernelCandidate("fused_tiled_16", "fused_tiled", enable_fusion=True, tile_m=16, tile_n=16, tile_k=16, opt_level="-O2"),
            # 8. Fused kernel with -O3
            KernelCandidate("fused_ikj_o3", "fused_ikj", enable_fusion=True, opt_level="-O3"),
        ]
        return candidates[:max_budget]

    def _get_compiler_sources_fingerprint(self) -> str:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sources = [
            "python/forgetune/autotuner.py",
            "python/forgetune/compiler.py",
            "python/forgetune/export_frontend.py",
            "src/codegen.cpp",
            "src/optimizer.cpp",
            "src/fusion.cpp",
            "src/memory_planner.cpp"
        ]
        h = hashlib.sha256()
        for s in sources:
            p = os.path.join(root, s)
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    h.update(f.read())
        return h.hexdigest()

    def _compute_cache_key(self, graph_json: str, input_shape: Tuple[int, ...], max_budget: int = 8) -> str:
        h = hashlib.sha256()
        h.update(b"forgetune_v6_reproducible_provenance_2.0_")
        h.update(graph_json.encode("utf-8"))
        h.update(str(input_shape).encode("utf-8"))
        h.update(str(max_budget).encode("utf-8"))
        h.update(f"atol_{self.atol:.1e}_rtol_{self.rtol:.1e}".encode("utf-8"))
        h.update(self._get_compiler_sources_fingerprint().encode("utf-8"))
        compiler_info = self.compiler.get_compiler_info()
        h.update(str(compiler_info.get("type", "")).encode("utf-8"))
        h.update(str(compiler_info.get("version", "")).encode("utf-8"))
        machine_meta = get_machine_metadata()
        h.update(str(machine_meta.get("processor", "")).encode("utf-8"))
        h.update(str(machine_meta.get("machine", "")).encode("utf-8"))
        return h.hexdigest()

    def select_best_candidate(
        self,
        verified_candidates: List[Tuple[KernelCandidate, Any, Dict[str, Any]]]
    ) -> Tuple[KernelCandidate, float, str]:
        """
        Applies statistically defensible, deterministic candidate selection and tie-breaking:
        1. If macro end-to-end improvement is statistically significant (> max(2.0, base_iqr))
           and not a micro-workload (< 1.0 us kernel), select fastest e2e candidate.
        2. Otherwise (differences are noise-dominated), select by reliable in-kernel latency:
           - Filter candidates improving over baseline kernel by >= 5%.
           - Group candidates within 5% of fastest in-kernel candidate (tie band).
           - Break ties in tie band by simplicity: prefer -O2 over -O3, prefer untiled over tiled.
           - If no candidate improves by >= 5%, retain baseline_naive.
        """
        if not verified_candidates:
            raise AutotuningError("Cannot select candidate from empty verified candidate list.")

        baseline_entry = next((cr for cr in verified_candidates if cr[0].candidate_id == "baseline_naive"), None)
        baseline_cand = baseline_entry[0] if baseline_entry else verified_candidates[0][0]
        base_median = (
            baseline_entry[2]["benchmark"]["median_latency_us"]
            if (baseline_entry and baseline_entry[2].get("benchmark"))
            else (verified_candidates[0][2].get("benchmark", {}).get("median_latency_us", 0.0))
        )
        base_kernel = baseline_entry[2].get("in_kernel_latency_us") if baseline_entry else None
        base_iqr = (
            baseline_entry[2]["benchmark"]["iqr_us"]
            if (baseline_entry and baseline_entry[2].get("benchmark"))
            else 1.0
        )

        best_candidate = baseline_cand
        best_latency = base_median
        selection_rationale = "Baseline candidate retained"

        candidates_with_kernel = [
            c for c in verified_candidates if c[2].get("in_kernel_latency_us") is not None
        ]

        if candidates_with_kernel and base_kernel and base_kernel > 0:
            # Enforce that candidates must show genuine >= 5% in-kernel speedup over baseline
            improving = [
                c for c in candidates_with_kernel
                if (base_kernel / c[2]["in_kernel_latency_us"]) >= 1.05
            ]
            if improving:
                improving.sort(key=lambda x: x[2]["in_kernel_latency_us"])
                fastest_k = improving[0][2]["in_kernel_latency_us"]

                # Group candidates within 5% of the fastest candidate (tie band)
                top_tier = [
                    c for c in improving
                    if (c[2]["in_kernel_latency_us"] <= fastest_k * 1.05)
                ]

                def candidate_simplicity_rank(cand_tuple):
                    cand, rt, res_entry = cand_tuple
                    opt_score = 0 if cand.opt_level == "-O2" else 1
                    tile_score = 0 if cand.tile_m is None else 1
                    return (opt_score, tile_score, res_entry["in_kernel_latency_us"])

                top_tier.sort(key=candidate_simplicity_rank)
                best_cand_tuple = top_tier[0]
                best_candidate = best_cand_tuple[0]
                best_k_res = best_cand_tuple[2]
                best_latency = (
                    best_k_res.get("benchmark", {}).get("median_latency_us", base_median)
                    if best_k_res.get("benchmark")
                    else base_median
                )
                k_ratio = base_kernel / best_k_res["in_kernel_latency_us"]

                simplicity_note = ""
                if best_candidate.opt_level == "-O2":
                    simplicity_note = "; simpler -O2 candidate selected over -O3 (no defensible -O3 advantage)"

                selection_rationale = (
                    f"selected {best_candidate.candidate_id} ({best_candidate.opt_level}) based on genuine {k_ratio:.2f}x in-kernel speedup "
                    f"({best_k_res['in_kernel_latency_us']*1000:.1f} ns vs {base_kernel*1000:.1f} ns baseline){simplicity_note}"
                )
            else:
                best_candidate = baseline_cand
                best_latency = base_median
                selection_rationale = "All candidate in-kernel differences within measurement noise (<5% speedup) or slower; simpler baseline preferred"
        else:
            # Fallback when in-kernel timings are not available (e.g. mock runtimes)
            candidates_by_e2e = sorted(
                verified_candidates,
                key=lambda x: x[2].get("benchmark", {}).get("median_latency_us", float("inf"))
                if x[2].get("benchmark")
                else float("inf")
            )
            fastest_e2e_cand, fastest_e2e_rt, fastest_e2e_res = candidates_by_e2e[0]
            fastest_e2e_lat = (
                fastest_e2e_res.get("benchmark", {}).get("median_latency_us", base_median)
                if fastest_e2e_res.get("benchmark")
                else base_median
            )
            e2e_diff = base_median - fastest_e2e_lat
            if base_median > 0 and (base_median / fastest_e2e_lat) >= 1.05 and e2e_diff > max(2.0, base_iqr):
                best_candidate = fastest_e2e_cand
                best_latency = fastest_e2e_lat
                selection_rationale = f"Significant end-to-end latency reduction ({e2e_diff:.2f} us delta exceeds noise window)"
            else:
                best_candidate = baseline_cand
                best_latency = base_median
                selection_rationale = "Differences within measurement noise; simpler baseline preferred"

        return best_candidate, best_latency, selection_rationale

    def tune(
        self,
        graph: Graph,
        example_input: np.ndarray,
        reference_output: np.ndarray,
        artifacts_dir: str,
        max_budget: int = 8,
        use_cache: bool = True,
        oracle: Optional[Any] = None
    ) -> Tuple[KernelCandidate, Dict[str, Any]]:
        """
        Runs correctness-gated autotuning across candidate configurations.
        Returns the winning KernelCandidate and the full tuning report.
        """
        if not isinstance(max_budget, int) or max_budget <= 0:
            raise ValueError(f"max_budget must be a positive integer, got {max_budget}")

        os.makedirs(artifacts_dir, exist_ok=True)
        graph_json = graph.to_json()
        input_shape = tuple(example_input.shape)
        output_shape = tuple(reference_output.shape)

        cache_key = self._compute_cache_key(graph_json, input_shape, max_budget=max_budget)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if use_cache and os.path.isfile(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_report = json.load(f)
                cand_data = cached_report["selected_candidate"]
                selected = KernelCandidate(**cand_data)
                valid_ids = {c.candidate_id for c in self.get_candidate_space(max_budget)}
                if selected.candidate_id in valid_ids:
                    return selected, cached_report
            except Exception:
                pass # Cache invalid/corrupted -> re-tune

        candidates = self.get_candidate_space(max_budget)
        candidate_results = []
        verified_candidates = []
        best_candidate = None
        best_latency = float("inf")

        # Create validation test inputs across representative batch sizes
        rng = np.random.default_rng(42)
        is_dyn = graph.has_dynamic_batch()
        max_b = getattr(graph, "max_batch_size", 128)
        if max_b <= 0:
            max_b = 128

        if is_dyn:
            b_example = input_shape[0] if input_shape[0] > 0 else 1
            test_batches = sorted(list({1, b_example, min(16, max_b), max_b}))
        else:
            test_batches = [input_shape[0] if input_shape[0] > 0 else 1]

        test_inputs = []
        for b in test_batches:
            shape_b = (b, *input_shape[1:])
            test_inputs.append(rng.standard_normal(shape_b).astype(np.float32))
            test_inputs.append(np.zeros(shape_b, dtype=np.float32))
            test_inputs.append(-np.abs(rng.standard_normal(shape_b).astype(np.float32)))
            test_inputs.append(rng.uniform(-5.0, 5.0, size=shape_b).astype(np.float32))

        test_inputs.insert(0, example_input)

        # Ground truth outputs from reference interpreter for all test inputs
        test_ground_truths = []
        for tin in test_inputs:
            ref_dict = graph.evaluate({graph.inputs[0]: tin})
            test_ground_truths.append(ref_dict[graph.outputs[0]])

        # Strict differential check: Verify interpreter ground truth matches PyTorch reference output on example_input
        if not np.allclose(test_ground_truths[0], reference_output, atol=self.atol, rtol=self.rtol):
            diff = float(np.max(np.abs(test_ground_truths[0] - reference_output)))
            raise CorrectnessError(f"Interpreter ground truth diverged from PyTorch reference output: max_abs={diff}")

        if oracle is not None:
            for tin, exp in zip(test_inputs, test_ground_truths):
                pt_out = oracle(tin)
                if not np.allclose(exp, pt_out, atol=self.atol, rtol=self.rtol):
                    diff = float(np.max(np.abs(exp - pt_out)))
                    raise CorrectnessError(f"Interpreter ground truth diverged from PyTorch oracle at batch {tin.shape[0]}: max_abs={diff}")

        from .compiler import get_engine_executable
        engine_exe = get_engine_executable()

        import subprocess

        for cand in candidates:
            res_entry = {
                "candidate": cand.to_dict(),
                "status": "pending",
                "rejection_reason": None,
                "benchmark": None
            }

            cand_dir = os.path.join(artifacts_dir, cand.candidate_id)
            os.makedirs(cand_dir, exist_ok=True)
            cand_json = os.path.join(cand_dir, "graph.json")
            cand_cpp = os.path.join(cand_dir, "kernel.cpp")
            cand_dll = os.path.join(cand_dir, "kernel.dll")

            with open(cand_json, "w", encoding="utf-8") as f:
                f.write(graph_json)

            # 1. Code Generation: Try in-process C++ compiler first, fallback to CLI
            codegen_done = False
            try:
                from . import _forgetune_cpp as cpp
                g_cpp = cpp.json_to_graph(graph_json)
                cfg = cpp.KernelConfig()
                cfg.candidate_id = cand.candidate_id
                variant_map = {
                    "naive_ijk": cpp.MatMulVariant.NAIVE_IJK,
                    "loop_ikj": cpp.MatMulVariant.LOOP_IKJ,
                    "tiled": cpp.MatMulVariant.TILED,
                    "fused_ikj": cpp.MatMulVariant.FUSED_IKJ,
                    "fused_tiled": cpp.MatMulVariant.FUSED_TILED
                }
                if cand.variant not in variant_map:
                    raise ValueError(f"Unknown candidate variant: {cand.variant}")
                cfg.matmul_variant = variant_map[cand.variant]
                cfg.enable_fusion = bool(cand.enable_fusion)
                if cand.tile_m is not None:
                    cfg.tile_m = int(cand.tile_m)
                    cfg.tile_n = int(cand.tile_n)
                    cfg.tile_k = int(cand.tile_k)
                compiler = cpp.Compiler(cfg)
                res = compiler.compile(g_cpp)
                if res.success:
                    with open(cand_cpp, "w", encoding="utf-8") as f:
                        f.write(res.generated_cpp_source)
                    codegen_done = True
                else:
                    res_entry["status"] = "codegen_failed"
                    res_entry["rejection_reason"] = res.error_message
                    candidate_results.append(res_entry)
                    continue
            except Exception:
                pass

            if not codegen_done:
                engine_cmd = [
                    engine_exe,
                    "--compile", cand_json,
                    "-o", cand_cpp,
                    "--variant", cand.variant,
                    "--candidate-id", cand.candidate_id
                ]
                if cand.tile_m is not None:
                    engine_cmd.extend([
                        "--tile-m", str(cand.tile_m),
                        "--tile-n", str(cand.tile_n),
                        "--tile-k", str(cand.tile_k)
                    ])
                if cand.enable_fusion:
                    engine_cmd.append("--fusion")

                try:
                    p = subprocess.run(engine_cmd, capture_output=True, text=True, timeout=15)
                    if p.returncode != 0:
                        res_entry["status"] = "codegen_failed"
                        res_entry["rejection_reason"] = p.stderr
                        candidate_results.append(res_entry)
                        continue
                except Exception as e:
                    res_entry["status"] = "codegen_error"
                    res_entry["rejection_reason"] = str(e)
                    candidate_results.append(res_entry)
                    continue

            # 2. Compile shared library
            try:
                comp_info = self.compiler.compile_shared_library(
                    cand_cpp, cand_dll, opt_level=cand.opt_level
                )
            except Exception as e:
                res_entry["status"] = "compilation_failed"
                res_entry["rejection_reason"] = str(e)
                candidate_results.append(res_entry)
                continue

            # 3. Correctness Gate
            try:
                is_dyn = graph.has_dynamic_batch()
                rt_in_shape = tuple(-1 if (is_dyn and i == 0) else s for i, s in enumerate(input_shape))
                out_node = graph.nodes[graph.outputs[0]]
                out_dyn = (len(out_node.shape) > 0 and out_node.shape[0] == -1)
                rt_out_shape = tuple(-1 if (out_dyn and i == 0) else s for i, s in enumerate(output_shape))
                runtime = NativeRuntime(cand_dll, rt_in_shape, rt_out_shape)
                for idx, (tin, expected) in enumerate(zip(test_inputs, test_ground_truths)):
                    actual = runtime.run(tin)
                    if actual.shape != expected.shape:
                        raise CorrectnessError(
                            f"Output shape mismatch for candidate {cand.candidate_id} at batch {tin.shape[0]}: "
                            f"expected {expected.shape}, got {actual.shape}"
                        )
                    max_abs = float(np.max(np.abs(actual - expected)))
                    max_rel = float(np.max(np.abs((actual - expected) / (np.abs(expected) + 1e-7))))
                    if not np.allclose(actual, expected, atol=self.atol, rtol=self.rtol):
                        raise CorrectnessError(
                            f"Numerical mismatch vs IR interpreter at batch {tin.shape[0]}: max_abs={max_abs}, max_rel={max_rel}",
                            max_abs_diff=max_abs,
                            max_rel_diff=max_rel
                        )
                    if oracle is not None:
                        pt_exp = oracle(tin)
                        if actual.shape != pt_exp.shape or not np.allclose(actual, pt_exp, atol=self.atol, rtol=self.rtol):
                            diff_pt = float(np.max(np.abs(actual - pt_exp)))
                            raise CorrectnessError(f"Numerical mismatch vs PyTorch oracle at batch {tin.shape[0]}: max_abs={diff_pt}")
                    elif idx == 0 and not np.allclose(actual, reference_output, atol=self.atol, rtol=self.rtol):
                        diff_pt = float(np.max(np.abs(actual - reference_output)))
                        raise CorrectnessError(f"Numerical mismatch vs PyTorch reference: max_abs={diff_pt}")
                res_entry["status"] = "passed"
                candidate_results.append(res_entry)
                verified_candidates.append((cand, runtime, res_entry))
            except Exception as e:
                res_entry["status"] = "correctness_failed"
                res_entry["rejection_reason"] = str(e)
                candidate_results.append(res_entry)
                continue

        if not verified_candidates:
            raise AutotuningError("Autotuner failed: no candidate passed compilation and correctness gate.")

        # Phase 2: Interleaved Multi-Round Benchmarking Across All Verified Candidates
        # 1. Warm-up all passing candidates equally to eliminate initial-candidate bias
        for _, rt, _ in verified_candidates:
            for _ in range(25):
                rt.run(example_input)

        # 2. Pure in-kernel isolated execution timing (3 interleaved rounds inside C++ via steady_clock)
        is_dyn = graph.has_dynamic_batch()
        if is_dyn:
            eval_batches = sorted(list({1, min(4, max_b), min(16, max_b), max_b}))
            eval_inputs = [
                np.ascontiguousarray(np.tile(example_input[:1], (b, *([1] * (example_input.ndim - 1)))))
                for b in eval_batches
            ]
            k_iters = 2000
        else:
            eval_inputs = [example_input]
            k_iters = 10000

        k_samples = {cand.candidate_id: [] for cand, _, _ in verified_candidates}
        for _ in range(3):
            for cand, rt, res_entry in verified_candidates:
                try:
                    total_k_lat = 0.0
                    for inp in eval_inputs:
                        total_k_lat += rt.benchmark_kernel(inp, iters=k_iters)
                    k_samples[cand.candidate_id].append(total_k_lat)
                except Exception as e:
                    res_entry["benchmark_warning"] = str(e)

        for cand, rt, res_entry in verified_candidates:
            samples = k_samples.get(cand.candidate_id, [])
            res_entry["in_kernel_latency_us"] = float(np.median(samples)) if samples else None

        # 3. Interleaved end-to-end Python invocation measurements
        NUM_ROUNDS = 5
        ITERS_PER_ROUND = 20
        e2e_samples = {cand.candidate_id: [] for cand, _, _ in verified_candidates}
        for round_idx in range(NUM_ROUNDS):
            for cand, rt, _ in verified_candidates:
                bm = benchmark_function(lambda: rt.run(example_input), warmup_iters=5, measure_iters=ITERS_PER_ROUND)
                e2e_samples[cand.candidate_id].extend(bm["raw_samples_us"])

        # Record rigorous summary metrics for each candidate
        for cand, rt, res_entry in verified_candidates:
            arr = np.array(e2e_samples[cand.candidate_id], dtype=np.float64)
            q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
            res_entry["benchmark"] = {
                "median_latency_us": float(np.median(arr)),
                "mean_latency_us": float(np.mean(arr)),
                "min_latency_us": float(np.min(arr)),
                "max_latency_us": float(np.max(arr)),
                "std_dev_us": float(np.std(arr)),
                "iqr_us": float(q75 - q25),
                "measured_iterations": len(arr),
                "in_kernel_latency_us": res_entry.get("in_kernel_latency_us")
            }

        # Phase 3: Statistically Defensible Candidate Selection
        best_candidate, best_latency, selection_rationale = self.select_best_candidate(verified_candidates)
        base_median = best_latency
        baseline_entry = next((cr for cr in verified_candidates if cr[0].candidate_id == "baseline_naive"), None)
        if baseline_entry and "benchmark" in baseline_entry[2] and baseline_entry[2]["benchmark"]:
            base_median = baseline_entry[2]["benchmark"]["median_latency_us"]

        speedup = (base_median / best_latency) if base_median and best_latency else 1.0

        tuning_report = {
            "schema_version": "2.0",
            "selected_candidate": best_candidate.to_dict(),
            "best_median_latency_us": best_latency,
            "baseline_median_latency_us": base_median,
            "measured_speedup": speedup,
            "selection_rationale": selection_rationale,
            "machine_metadata": get_machine_metadata(),
            "candidate_evaluations": candidate_results
        }

        # Save report and cache entry atomically
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            write_atomic_json(cache_file, tuning_report)

        report_path = os.path.join(artifacts_dir, "autotuning_report.json")
        write_atomic_json(report_path, tuning_report)

        return best_candidate, tuning_report
