"""
Acceptance Test Suite: Autotuner Candidate Space, Gating & Selection Verification
Challenges the autotuner's candidate generation, correctness-gating mechanics,
statistical tie-breaking policy, and caching resilience against corrupted data.
"""

import os
import pytest
import numpy as np

from forgetune.autotuner import Autotuner, KernelCandidate
from forgetune.graph_adapter import Graph, Node

@pytest.fixture
def temp_autotune_dir(tmp_path):
    d = os.path.abspath(str(tmp_path / "acceptance_tuner"))
    os.makedirs(d, exist_ok=True)
    yield d

def test_acceptance_candidate_space_coverage():
    """
    Verifies that the autotuner candidate generator produces valid configurations
    spanning all 5 kernel variants, tile sizes (8, 16), and optimization flags (-O2, -O3).
    """
    tuner = Autotuner()
    candidates = tuner.get_candidate_space(max_budget=8)
    assert len(candidates) == 8

    variants = {c.variant for c in candidates}
    expected_variants = {"naive_ijk", "loop_ikj", "tiled", "fused_ikj", "fused_tiled"}
    assert variants == expected_variants, f"Missing kernel variants in search space: {expected_variants - variants}"

    # Check tile sizes present
    tiled_cands = [c for c in candidates if c.tile_m is not None]
    tile_sizes = {(c.tile_m, c.tile_n, c.tile_k) for c in tiled_cands}
    assert (8, 8, 8) in tile_sizes
    assert (16, 16, 16) in tile_sizes

    # Check opt levels present
    opt_levels = {c.opt_level for c in candidates}
    assert "-O2" in opt_levels
    assert "-O3" in opt_levels

def test_acceptance_correctness_gating_rejects_failing_candidate(temp_autotune_dir, monkeypatch):
    """
    Challenges correctness gating: if a candidate yields erroneous numerical output,
    the autotuner must immediately reject that candidate, mark it as 'correctness_failed',
    and never select it as the winning configuration.
    """
    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", "X", shape=[2, 4], dtype="f32"))
    w_id = g.add_node(Node(1, "CONST", "W", shape=[4, 4], dtype="f32", const_data=[1.0]*16))
    mm_id = g.add_node(Node(2, "MATMUL", "mm", inputs=[in_id, w_id], shape=[2, 4], dtype="f32"))
    out_id = g.add_node(Node(3, "OUTPUT", "out", inputs=[mm_id], shape=[2, 4], dtype="f32"))
    g.set_inputs([in_id])
    g.set_outputs([out_id])

    x_val = np.ones((2, 4), dtype=np.float32)
    ref_out = np.full((2, 4), 4.0, dtype=np.float32)

    # Patch NativeRuntime.run for candidate 'loop_ikj' to inject numerical errors
    from forgetune.runtime import NativeRuntime
    real_run = NativeRuntime.run

    def poisoned_run(self, input_tensor):
        res = real_run(self, input_tensor)
        # Poison if the DLL corresponds to loop_ikj
        if "loop_ikj" in self.original_dll_path:
            return res + 999.0
        return res

    monkeypatch.setattr(NativeRuntime, "run", poisoned_run)

    cache_dir = os.path.join(temp_autotune_dir, "cache")
    tuner = Autotuner(cache_dir=cache_dir)
    selected_cand, report = tuner.tune(
        graph=g,
        example_input=x_val,
        reference_output=ref_out,
        artifacts_dir=temp_autotune_dir,
        max_budget=4,
        use_cache=False
    )

    # loop_ikj must have been rejected!
    evals = report["candidate_evaluations"]
    loop_ikj_eval = next((e for e in evals if e["candidate"]["candidate_id"] == "loop_ikj"), None)
    assert loop_ikj_eval is not None
    assert loop_ikj_eval["status"] == "correctness_failed"
    assert "Numerical mismatch" in loop_ikj_eval["rejection_reason"]

    # Selected candidate must not be loop_ikj
    assert selected_cand.candidate_id != "loop_ikj"

def test_acceptance_autotuning_cache_hit_and_corruption_resilience(temp_autotune_dir):
    """
    Tests autotuner cache mechanics:
    1. Valid cache hits bypass recompilation and return the cached plan.
    2. Corrupted cache files (malformed JSON or invalid schema) gracefully trigger re-tuning.
    """
    g = Graph()
    in_id = g.add_node(Node(0, "INPUT", "X", shape=[2, 4], dtype="f32"))
    w_id = g.add_node(Node(1, "CONST", "W", shape=[4, 4], dtype="f32", const_data=[0.5]*16))
    mm_id = g.add_node(Node(2, "MATMUL", "mm", inputs=[in_id, w_id], shape=[2, 4], dtype="f32"))
    out_id = g.add_node(Node(3, "OUTPUT", "out", inputs=[mm_id], shape=[2, 4], dtype="f32"))
    g.set_inputs([in_id])
    g.set_outputs([out_id])

    x_val = np.ones((2, 4), dtype=np.float32)
    ref_out = np.full((2, 4), 2.0, dtype=np.float32)

    cache_dir = os.path.join(temp_autotune_dir, "cache")
    tuner = Autotuner(cache_dir=cache_dir)

    # First run: tunes and saves cache
    cand1, report1 = tuner.tune(
        graph=g,
        example_input=x_val,
        reference_output=ref_out,
        artifacts_dir=os.path.join(temp_autotune_dir, "run1"),
        max_budget=3,
        use_cache=True
    )

    # Verify cache file was written
    cache_files = os.listdir(cache_dir)
    assert len(cache_files) == 1
    cache_path = os.path.join(cache_dir, cache_files[0])

    # Second run: cache hit
    cand2, report2 = tuner.tune(
        graph=g,
        example_input=x_val,
        reference_output=ref_out,
        artifacts_dir=os.path.join(temp_autotune_dir, "run2"),
        max_budget=3,
        use_cache=True
    )
    assert cand2.candidate_id == cand1.candidate_id

    # 3. Corrupt cache file with garbage content
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write("{ INVALID JSON DATA !@#$%^&*")

    # Third run: must not crash, must fall back to fresh tuning and rewrite cache
    cand3, report3 = tuner.tune(
        graph=g,
        example_input=x_val,
        reference_output=ref_out,
        artifacts_dir=os.path.join(temp_autotune_dir, "run3"),
        max_budget=3,
        use_cache=True
    )
    assert cand3 is not None
    assert "selected_candidate" in report3

def _make_mock_candidate(
    candidate_id: str,
    variant: str = "naive_ijk",
    opt_level: str = "-O2",
    tile_m = None,
    median_e2e_us: float = 10.0,
    iqr_e2e_us: float = 1.0,
    in_kernel_us: float = 0.05
):
    cand = KernelCandidate(
        candidate_id=candidate_id,
        variant=variant,
        enable_fusion=variant.startswith("fused"),
        tile_m=tile_m,
        tile_n=tile_m,
        tile_k=tile_m,
        opt_level=opt_level
    )
    res_entry = {
        "candidate": cand.to_dict(),
        "benchmark": {
            "median_latency_us": median_e2e_us,
            "iqr_us": iqr_e2e_us,
        },
        "in_kernel_latency_us": in_kernel_us
    }
    return (cand, None, res_entry)

def test_acceptance_autotuner_baseline_fallback_on_noise_band():
    """
    Verifies that when all alternative candidates perform within the <5% noise band
    of the baseline kernel, the autotuner strictly retains 'baseline_naive'.
    """
    tuner = Autotuner()
    base = _make_mock_candidate("baseline_naive", "naive_ijk", "-O2", in_kernel_us=0.050)
    c1 = _make_mock_candidate("loop_ikj", "loop_ikj", "-O2", in_kernel_us=0.048)  # 1.041x speedup (< 1.05x)
    c2 = _make_mock_candidate("fused_ikj", "fused_ikj", "-O2", in_kernel_us=0.049) # 1.020x speedup (< 1.05x)

    selected, lat, rationale = tuner.select_best_candidate([base, c1, c2])
    assert selected.candidate_id == "baseline_naive"
    assert "simpler baseline preferred" in rationale

def test_acceptance_autotuner_selects_faster_candidate_above_threshold():
    """
    Verifies that when a candidate genuinely achieves >= 5% in-kernel speedup,
    the autotuner selects it over baseline_naive.
    """
    tuner = Autotuner()
    base = _make_mock_candidate("baseline_naive", "naive_ijk", "-O2", in_kernel_us=0.050)
    c1 = _make_mock_candidate("loop_ikj", "loop_ikj", "-O2", in_kernel_us=0.040)  # 1.25x speedup (>= 1.05x)

    selected, lat, rationale = tuner.select_best_candidate([base, c1])
    assert selected.candidate_id == "loop_ikj"
    assert "selected loop_ikj" in rationale

def test_acceptance_autotuner_tie_breaking_prefers_o2_over_o3():
    """
    Verifies the tie-breaking rule: when -O2 and -O3 candidates are within 5% of
    the fastest candidate, the autotuner prefers the simpler -O2 candidate and
    explicitly documents the rationale.
    """
    tuner = Autotuner()
    base = _make_mock_candidate("baseline_naive", "naive_ijk", "-O2", in_kernel_us=0.100)
    c_o3 = _make_mock_candidate("fused_ikj_o3", "fused_ikj", "-O3", in_kernel_us=0.050)
    c_o2 = _make_mock_candidate("fused_ikj", "fused_ikj", "-O2", in_kernel_us=0.051) # 0.051 <= 0.050 * 1.05 = 0.0525

    selected, lat, rationale = tuner.select_best_candidate([base, c_o3, c_o2])
    assert selected.candidate_id == "fused_ikj"
    assert selected.opt_level == "-O2"
    assert "simpler -O2 candidate selected over -O3" in rationale

def test_acceptance_autotuner_tie_breaking_prefers_untiled_over_tiled():
    """
    Verifies that when an untiled candidate and a tiled candidate are tied within 5%,
    the simpler untiled candidate is preferred.
    """
    tuner = Autotuner()
    base = _make_mock_candidate("baseline_naive", "naive_ijk", "-O2", in_kernel_us=0.100)
    c_untiled = _make_mock_candidate("loop_ikj", "loop_ikj", "-O2", tile_m=None, in_kernel_us=0.050)
    c_tiled = _make_mock_candidate("tiled_8", "tiled", "-O2", tile_m=8, in_kernel_us=0.050)

    selected, lat, rationale = tuner.select_best_candidate([base, c_tiled, c_untiled])
    assert selected.candidate_id == "loop_ikj"
    assert selected.tile_m is None

def test_acceptance_autotuner_rejects_slower_candidates():
    """
    Verifies that candidates that run slower than baseline are never selected,
    and baseline_naive is safely retained.
    """
    tuner = Autotuner()
    base = _make_mock_candidate("baseline_naive", "naive_ijk", "-O2", in_kernel_us=0.050)
    c_slow = _make_mock_candidate("tiled_16", "tiled", "-O2", tile_m=16, in_kernel_us=0.090)

    selected, lat, rationale = tuner.select_best_candidate([base, c_slow])
    assert selected.candidate_id == "baseline_naive"
    assert "simpler baseline preferred" in rationale

def test_acceptance_autotuner_cache_key_shape_sensitivity():
    """
    Verifies that cache keys strictly differentiate input shapes and are 100% deterministic.
    """
    tuner = Autotuner()
    dummy_graph = '{"nodes": [{"id": 0, "op_type": "INPUT"}]}'
    k_batch2 = tuner._compute_cache_key(dummy_graph, (2, 16))
    k_batch4 = tuner._compute_cache_key(dummy_graph, (4, 16))
    k_batch2_repeat = tuner._compute_cache_key(dummy_graph, (2, 16))

    assert k_batch2 != k_batch4, "Cache key failed to distinguish input batch sizes!"
    assert k_batch2 == k_batch2_repeat, "Cache key calculation is non-deterministic!"

def test_acceptance_autotuner_deterministic_tie_breaking_repeatability():
    """
    Verifies that 50 consecutive selection runs on identical candidate sets
    yield 100% identical winner IDs and rationale strings.
    """
    tuner = Autotuner()
    base = _make_mock_candidate("baseline_naive", "naive_ijk", "-O2", in_kernel_us=0.100)
    c1 = _make_mock_candidate("fused_ikj", "fused_ikj", "-O2", in_kernel_us=0.051)
    c2 = _make_mock_candidate("fused_ikj_o3", "fused_ikj", "-O3", in_kernel_us=0.050)
    c3 = _make_mock_candidate("loop_ikj", "loop_ikj", "-O2", in_kernel_us=0.055)

    ref_selected, ref_lat, ref_rat = tuner.select_best_candidate([base, c1, c2, c3])
    for _ in range(50):
        s, l, r = tuner.select_best_candidate([base, c1, c2, c3])
        assert s.candidate_id == ref_selected.candidate_id
        assert r == ref_rat

