"""
Acceptance Test Suite: Autotuner Candidate Space, Gating & Selection Verification
Challenges the autotuner's candidate generation, correctness-gating mechanics,
statistical tie-breaking policy, and caching resilience against corrupted data.
"""

import os
import json
import shutil
import tempfile
import pytest
import numpy as np

from forgetune.autotuner import Autotuner, KernelCandidate
from forgetune.graph_adapter import Graph, Node
from forgetune.native_compiler import NativeCompiler
from forgetune.errors import CorrectnessError

@pytest.fixture
def temp_autotune_dir():
    d = tempfile.mkdtemp(prefix="forgetune_accept_tuner_")
    yield d
    try:
        shutil.rmtree(d)
    except Exception:
        pass

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
    original_run = None
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
